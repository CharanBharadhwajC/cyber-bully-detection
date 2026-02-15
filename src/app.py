# src/app.py
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from flask import (
    Flask, render_template, request, g, jsonify,
    session, redirect, url_for
)

from utils import predict_text, MODELS_DIR

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE = Path(__file__).resolve().parent
TEMPLATES = BASE.parent / "templates"
STATIC = BASE.parent / "static"
DB_PATH = BASE / "database.db"

# ---------------------------------------------------
# App Initialization
# ---------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(TEMPLATES),
    static_folder=str(STATIC)
)

# Secure secret key for production
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Session Hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

# ---------------------------------------------------
# Database Handling
# ---------------------------------------------------
def get_db():
    """Return active SQLite connection."""
    db = getattr(g, "_database", None)
    if db is None:
        db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        db.row_factory = sqlite3.Row
        g._database = db
    return db


@app.teardown_appcontext
def close_db(exc):
    """Close DB connection on teardown."""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    """Initialize SQLite DB and create flagged_messages table."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS flagged_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            message TEXT,
            label TEXT,
            confidence REAL,
            rationale TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """Main chat route."""
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Store user input
            session["chat_history"].append({
                "sender": "user",
                "text": message
            })

            # Predict text classification
            try:
                pred = predict_text(message)
            except Exception as e:
                pred = {
                    "label": "error",
                    "confidence": 0.0,
                    "model": f"error: {str(e)}"
                }

            label = pred.get("label", "unknown")
            confidence = float(pred.get("confidence", 0))
            model_name = pred.get("model", "unknown")

            # Create rationale
            rationale = f"{model_name} predicted {label} (p={confidence:.2f})"

            # Format chatbot response
            response = (
                f"<strong>Result:</strong> {label.upper()}<br>"
                f"<small>Confidence: {confidence*100:.2f}%</small><br>"
                f"<small>{rationale}</small>"
            )

            # Add bot reply
            session["chat_history"].append({
                "sender": "bot",
                "text": response
            })

            # Store flagged messages
            if label in ("offensive", "hate_speech"):
                db = get_db()
                db.execute(
                    """
                    INSERT INTO flagged_messages
                    (model, message, label, confidence, rationale)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_name, message, label, confidence, rationale)
                )
                db.commit()

            session.modified = True

        return redirect(url_for("index"))

    return render_template(
        "index.html",
        chat_history=session.get("chat_history", [])
    )


@app.route("/clear_chat")
def clear_chat():
    """Clear current chat session."""
    session.pop("chat_history", None)
    return redirect(url_for("index"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API for text classification."""
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("message") or payload.get("text") or ""

    if not text:
        return jsonify({"error": "Empty message"}), 400

    try:
        pred = predict_text(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(pred)


@app.route("/admin")
def admin():
    """Admin dashboard: view flagged messages (converted to IST)."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM flagged_messages ORDER BY id DESC LIMIT 500"
    ).fetchall()

    # Convert UTC → IST
    ist_rows = []

    for r in rows:
        record = dict(r)

        if record.get("created_at"):
            try:
                utc_time = datetime.strptime(
                    record["created_at"],
                    "%Y-%m-%d %H:%M:%S"
                )
                ist_time = utc_time + timedelta(hours=5, minutes=30)
                record["created_at"] = ist_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                pass

        ist_rows.append(record)

    return render_template("admin.html", rows=ist_rows)


@app.route("/stats")
def stats():
    """Show stats for all flagged messages."""
    db = get_db()

    rows = db.execute(
        "SELECT label, COUNT(*) AS cnt FROM flagged_messages GROUP BY label"
    ).fetchall()

    counts = {r["label"]: r["cnt"] for r in rows}

    model_name = os.path.basename(str(MODELS_DIR))

    return render_template(
        "stats.html",
        counts=counts,
        model_name=model_name
    )


# ---------------------------------------------------
# Production Entry
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

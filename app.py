from flask import Flask, request, jsonify, Response
import os, pickle, traceback
from groq import Groq
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import get_categories, route_question, build_category_embeddings
from immutable_facts import load_facts, check_immutable

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")
HIERARCHY_FOLDER = "nli_hierarchy_v7"
TOP_K_NODES = 7

app = Flask(__name__)
groq_client = Groq(api_key=GROQ_API_KEY)

print("Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.\n")
print("Loading immutable facts...")
load_facts(embedder)
print("Building category embeddings from hierarchies...")
build_category_embeddings(HIERARCHY_FOLDER)
print("Done.\n")

_hierarchy_cache = {}
_node_embedding_cache = {}

def load_hierarchy(category):
    if category in _hierarchy_cache:
        return _hierarchy_cache[category]
    for ext in [".pkl", ".pickle", ""]:
        path = os.path.join(HIERARCHY_FOLDER, category + ext)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    G = pickle.load(f)
                _hierarchy_cache[category] = G
                return G
            except Exception:
                pass
    _hierarchy_cache[category] = None
    return None

def retrieve_nodes(G, question, top_k=TOP_K_NODES):
    if G is None:
        return []
    nodes_info = []
    for nid in G.nodes:
        if nid != "ROOT" and G.nodes[nid].get("text"):
            nodes_info.append((nid, G.nodes[nid]["text"], G.nodes[nid].get("level", 1)))
    if not nodes_info:
        return []
    node_ids, node_texts, node_levels = zip(*nodes_info)
    graph_id = id(G)
    if graph_id not in _node_embedding_cache:
        _node_embedding_cache[graph_id] = embedder.encode(
            list(node_texts), convert_to_tensor=True, show_progress_bar=False
        )
    node_embeddings = _node_embedding_cache[graph_id]
    q_emb = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, node_embeddings)[0].tolist()
    adjusted = [(s + (0.08 if nid.startswith("parent_") else 0.0), txt)
                for s, txt, nid in zip(scores, node_texts, node_ids)]
    adjusted.sort(reverse=True)
    seen = set()
    top_rules = []
    for _, text in adjusted:
        key = text[:80]
        if key not in seen:
            seen.add(key)
            top_rules.append(text)
        if len(top_rules) >= top_k:
            break
    return top_rules

def get_answer(question, context_rules):
    context_text = "\n".join("- " + r for r in context_rules)
    prompt = (
        "You are a knowledgeable assistant specializing in Indian Central Government service rules and policies.\n\n"
        "Rules from official DoPT documents:\n" + context_text +
        "\n\nAnswer this question clearly and concisely. Use 1-2 sentences for simple factual questions, "
        "and 3-5 sentences for complex questions that require explaining conditions or exceptions. "
        "Do not repeat information. Start with the direct answer. Do not mention OM numbers.\n\n"
        "Question: " + question
    )
    r = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=400
    )
    return r.choices[0].message.content.strip()

def get_answer_no_context(question):
    prompt = (
        "You are a helpful assistant for anyone with queries about Indian Central Government service rules and policies. "
        "If the message is a greeting, respond with just: Hi! How can I help you? "
        "Otherwise answer clearly and concisely. Use 1-2 sentences for simple factual questions, "
        "3-5 sentences for complex ones. Do not repeat information.\n\n"
        "Question: " + question
    )
    r = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=400
    )
    return r.choices[0].message.content.strip()


@app.route("/")
def index():
    return Response(get_html(), mimetype="text/html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"type": "answer", "answer": "Please enter a question.", "category": ""})

    
    immutable_answer = check_immutable(question, embedder)
    if immutable_answer:
        return jsonify({"type": "answer", "answer": immutable_answer, "category": "immutable"})

    categories = get_categories()
    category = route_question(question, categories)

    if category == "__clarify__":
        answer = get_answer_no_context(question)
        return jsonify({"type": "answer", "answer": answer, "category": "general"})

    try:
        G = load_hierarchy(category)
        if G and len(G.nodes) > 2:
            rules = retrieve_nodes(G, question)
            if rules:
                answer = get_answer(question, rules)
            else:
                answer = "I could not find a clear policy match for that query. Could you be more specific?"
        else:
            answer = get_answer_no_context(question)
        return jsonify({"type": "answer", "answer": answer, "category": category})
    except Exception:
        traceback.print_exc()
        return jsonify({"type": "error", "answer": "An error occurred. Please try again."})


def get_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DoPT Policy Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0a;--surface:#111;--border:#252525;--gold:#c9a84c;--gold-dim:#7a6030;--text:#e8e4dc;--dim:#6b6560;--muted:#333}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{flex-shrink:0;border-bottom:1px solid var(--border);padding:18px 36px;display:flex;align-items:center;gap:14px}
.logo{width:34px;height:34px;border:1px solid var(--gold);display:flex;align-items:center;justify-content:center;color:var(--gold);font-size:16px}
.ht{font-family:'Cormorant Garamond',serif;font-size:19px;font-weight:500}
.hs{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-top:2px}
#chatbox{flex:1;overflow-y:auto;padding:30px 36px;display:flex;flex-direction:column;gap:24px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
#chatbox::-webkit-scrollbar{width:3px}
#chatbox::-webkit-scrollbar-thumb{background:var(--border)}
#welcome{margin:auto;text-align:center;padding:40px 20px;max-width:480px}
.wi{font-family:'Cormorant Garamond',serif;font-size:52px;color:var(--gold);opacity:0.5;margin-bottom:18px}
.wt{font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:400}
.msg{display:flex;flex-direction:column;gap:6px}
.msg.user{align-self:flex-end;align-items:flex-end;max-width:60%}
.msg.bot{align-self:flex-start;align-items:flex-start;max-width:82%}
.msg.sys{align-self:flex-start;align-items:flex-start;max-width:70%}
.ml{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted)}
.msg.user .ml{color:var(--gold-dim)}
.bub{padding:14px 18px;font-size:13px;line-height:1.85}
.msg.user .bub{background:#141414;border:1px solid var(--border);border-right:2px solid var(--gold)}
.msg.bot .bub{background:var(--surface);border:1px solid var(--border);border-left:2px solid var(--gold-dim)}
.msg.sys .bub{background:rgba(201,168,76,0.04);border:1px solid rgba(201,168,76,0.15);color:#a89060;font-size:12px}
.dots{display:inline-flex;gap:4px;align-items:center;padding:4px 0}
.dots span{width:4px;height:4px;background:var(--gold);border-radius:50%;animation:blink 1.2s infinite;opacity:0.3}
.dots span:nth-child(2){animation-delay:0.2s}
.dots span:nth-child(3){animation-delay:0.4s}
@keyframes blink{0%,100%{opacity:0.3}50%{opacity:1}}
.iw{flex-shrink:0;border-top:1px solid var(--border);padding:18px 36px}
.ir{display:flex;align-items:center;border:1px solid var(--border);background:var(--surface)}
.ir:focus-within{border-color:var(--gold-dim)}
.ipfx{padding:0 12px;font-size:13px;color:var(--gold);opacity:0.5}
#qi{flex:1;background:transparent;border:none;outline:none;font-size:13px;font-family:'DM Mono',monospace;color:var(--text);padding:15px 0}
#qi::placeholder{color:var(--muted)}
#sb{background:transparent;border:none;border-left:1px solid var(--border);color:var(--dim);padding:0 20px;height:50px;cursor:pointer;font-size:10px;letter-spacing:2px;text-transform:uppercase;transition:all 0.2s}
#sb:hover{color:var(--gold)}
#sb:disabled{color:var(--muted);cursor:not-allowed}
</style>
</head>
<body>
<header>
  <div class="logo">&#9878;</div>
  <div>
    <div class="ht">DoPT Policy Assistant</div>
    <div class="hs">Department of Personnel &amp; Training</div>
  </div>
</header>
<div id="chatbox">
  <div id="welcome">
    <div class="wi">&#167;</div>
    <div class="wt">Central Government Service Rules</div>
  </div>
</div>
<div class="iw">
  <div class="ir">
    <span class="ipfx">&gt;_</span>
    <input type="text" id="qi" placeholder="Ask about DoPT service rules..." autocomplete="off">
    <button id="sb">Send</button>
  </div>
</div>
<script>
var qi = document.getElementById("qi");
var sb = document.getElementById("sb");
var cb = document.getElementById("chatbox");
sb.onclick = send;
qi.onkeydown = function(e) { if (e.key === "Enter") { e.preventDefault(); send(); } };
qi.focus();
function mk(tag, cls) { var el = document.createElement(tag); if (cls) el.className = cls; return el; }
function addMsg(type, lbl, fn) {
  var m = mk("div", "msg " + type);
  var l = mk("span", "ml"); l.textContent = lbl;
  var b = mk("div", "bub"); fn(b);
  m.appendChild(l); m.appendChild(b);
  cb.appendChild(m); cb.scrollTop = cb.scrollHeight;
  return m;
}
function send() {
  var q = qi.value.trim();
  if (!q || sb.disabled) return;
  var w = document.getElementById("welcome");
  if (w) w.parentNode.removeChild(w);
  sb.disabled = true; qi.value = "";
  addMsg("user", "You", function(b) { b.textContent = q; });
  var ld = addMsg("bot", "Assistant", function(b) {
    var d = mk("div", "dots");
    d.appendChild(mk("span")); d.appendChild(mk("span")); d.appendChild(mk("span"));
    b.appendChild(d);
  });
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/ask");
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onload = function() {
    if (ld.parentNode) ld.parentNode.removeChild(ld);
    if (xhr.status === 200) {
      var data = JSON.parse(xhr.responseText);
      if (data.type === "answer") {
        addMsg("bot", "Assistant", function(b) {
          var ans = mk("div"); ans.textContent = data.answer;
          b.appendChild(ans);
        });
      } else {
        addMsg("sys", "Note", function(b) { b.textContent = data.answer; });
      }
    } else {
      addMsg("sys", "Error", function(b) { b.textContent = "Something went wrong. Please try again."; });
    }
    sb.disabled = false; qi.focus(); cb.scrollTop = cb.scrollHeight;
  };
  xhr.onerror = function() {
    if (ld.parentNode) ld.parentNode.removeChild(ld);
    addMsg("sys", "Error", function(b) { b.textContent = "Connection error. Please try again."; });
    sb.disabled = false; qi.focus();
  };
  xhr.send(JSON.stringify({question: q}));
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
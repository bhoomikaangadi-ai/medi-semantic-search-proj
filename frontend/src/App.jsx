
import { useEffect, useState } from 'react';
import {
  Activity,
  Bot,
  Database,
  FileSearch,
  HeartPulse,
  LayoutDashboard,
  MessageCircle,
  Network,
  Send,
  ShieldCheck,
  Sparkles,
  GitBranch,
  Search,
  CheckCircle2,
  Menu
} from 'lucide-react';

const demoMetrics = [
  ['534', 'Source chunks', 'Current source dataset'],
  ['22,292', 'Candidate mentions', 'Detected by SciSpaCy'],
  ['4,810', 'Accepted mentions', 'Passed configured filters'],
  ['810', 'Unique entities', 'Extracted ontology entities']
];

const WELCOME_MESSAGE =
  'Welcome to MedGraph AI! Ask a medical question to explore answers and their supporting source documents.';

// Display source passages returned by FastAPI.
function MessageSources({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return null;
  }

  return (
    <div className="answer-sources">
      <h4>Sources used ({sources.length})</h4>

      {sources.map((source, index) => (
        <details
          className="source-card"
          key={`${source.chunk_id ?? index}-${index}`}
        >
          <summary>
            {source.title ||
              source.document ||
              source.source ||
              'Medical document'}
            {' — Chunk '}
            {source.chunk_id ?? 'N/A'}
          </summary>

          <div className="source-content">
            {source.document && (
              <p>
                <strong>Document:</strong> {source.document}
              </p>
            )}

            {source.source && (
              <p>
                <strong>Source:</strong> {source.source}
              </p>
            )}

            <p className="source-text">
              {source.text || 'No passage text available.'}
            </p>

            {source.score != null &&
              Number.isFinite(Number(source.score)) && (
                <small>
                  Similarity score: {Number(source.score).toFixed(4)}
                </small>
              )}

            {Array.isArray(source.concepts) &&
              source.concepts.length > 0 && (
                <p className="source-concepts">
                  <strong>Related concepts:</strong>{' '}
                  {source.concepts.join(', ')}
                </p>
              )}
          </div>
        </details>
      ))}
    </div>
  );
}

// Display AI-generated medical illustrations below the answer.
// Display medical illustrations below the answer.
function MessageImages({ images }) {
  if (!Array.isArray(images) || images.length === 0) {
    return null;
  }

  return (
    <div className="generated-images">
      <h4>Medical Illustrations</h4>

      <div className="generated-image-grid">
        {images.map((image, index) => {
          const imageUrl =
            typeof image === 'string'
              ? image
              : image?.image_url || image?.url;

          if (!imageUrl) {
            return null;
          }

          const description =
            typeof image === 'object'
              ? image.description || 'Educational medical illustration'
              : 'Educational medical illustration';

          return (
            <figure key={`${imageUrl}-${index}`}>
              <img
                src={imageUrl}
                alt={description}
                loading="lazy"
                onError={event => {
                  console.error('Unable to load generated image:', imageUrl);
                  event.currentTarget.style.display = 'none';
                }}
              />

              <figcaption>
                {description}
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState('Checking');
  const [page, setPage] = useState('Overview');
  const [entities, setEntities] = useState([]);
  const [term, setTerm] = useState('');
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: WELCOME_MESSAGE,
      sources: [],
      concepts: [],
      images: []
    }
  ]);

  // Check backend connection and load medical entities.
  useEffect(() => {
    fetch('/api/health')
      .then(async response => {
        if (!response.ok) {
          throw new Error('Health request failed');
        }

        return response.json();
      })
      .then(data => {
        setHealth(data.status === 'ok' ? 'Connected' : 'Check API');
      })
      .catch(() => {
        setHealth('Offline');
      });

    fetch('/api/entities?limit=200')
      .then(async response => {
        if (!response.ok) {
          throw new Error('Entity request failed');
        }

        return response.json();
      })
      .then(data => {
        setEntities(
          Array.isArray(data.entities) ? data.entities : []
        );
      })
      .catch(() => {
        setEntities([]);
      });
  }, []);

  // Send a question to the FastAPI backend.
  async function ask(text = q) {
    const question = text.trim();

    if (!question || busy) {
      return;
    }

    setMessages(current => [
      ...current,
      {
        role: 'user',
        text: question
      }
    ]);

    setQ('');
    setBusy(true);

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify({ question })
      });

      const raw = await response.text();

      if (!raw.trim()) {
        throw new Error(
          `The backend returned an empty response (HTTP ${response.status}). Check the FastAPI terminal.`
        );
      }

      let data;

      try {
        data = JSON.parse(raw);
      } catch {
        throw new Error(
          `The backend returned invalid JSON (HTTP ${response.status}): ${raw.slice(0, 250)}`
        );
      }

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed (HTTP ${response.status}).`
        );
      }

      // Support standard FastAPI and common wrapped responses.
      const payload =
        data?.data && typeof data.data === 'object'
          ? data.data
          : data?.result && typeof data.result === 'object'
            ? data.result
            : data?.response &&
                typeof data.response === 'object'
              ? data.response
              : data;

      const sources = Array.isArray(payload?.sources)
        ? payload.sources
        : Array.isArray(data?.sources)
          ? data.sources
          : [];

      const concepts = Array.isArray(payload?.concepts)
        ? payload.concepts
        : Array.isArray(data?.concepts)
          ? data.concepts
          : [];

      // NEW: Read generated images from the backend response.
      const images = Array.isArray(payload?.images)
        ? payload.images
        : Array.isArray(data?.images)
          ? data.images
          : [];

      const answer =
        payload?.answer ||
        data?.answer ||
        (typeof data?.response === 'string'
          ? data.response
          : '') ||
        'No answer returned.';

      // Save the complete answer, sources, concepts and images.
      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          text: answer,
          warning: payload?.warning || data?.warning || '',
          sources,
          concepts,
          images,
          model: payload?.model || data?.model || '',
          retrievedCount:
            payload?.retrieved_count ??
            data?.retrieved_count ??
            sources.length,
          responseKeys: Object.keys(data || {})
        }
      ]);
    } catch (error) {
      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          text: error.message || 'Unable to contact the backend.',
          warning: 'Backend request failed.',
          sources: [],
          concepts: [],
          images: [],
          isError: true
        }
      ]);
    } finally {
      setBusy(false);
    }
  }

  // Get a readable name for each medical entity.
  const name = entity =>
    typeof entity === 'string'
      ? entity
      : entity?.name ||
        entity?.label ||
        entity?.canonical_name ||
        entity?.entity ||
        'Medical entity';

  const filteredEntities = entities.filter(entity =>
    name(entity).toLowerCase().includes(term.toLowerCase())
  );

  return (
    <div className="layout">
      <aside className="side">
        <div className="logo">
          <span>
            <HeartPulse />
          </span>

          <div>
            <b>MedGraph AI</b>
            <small>Medical Graph RAG</small>
          </div>
        </div>

        <p className="label">WORKSPACE</p>

        {[
          ['Overview', LayoutDashboard],
          ['Medical AI Chat', MessageCircle],
          ['Entity Explorer', Network],
          ['Pipeline & Tools', GitBranch]
        ].map(([label, Icon]) => (
          <button
            className={`nav ${page === label ? 'active' : ''}`}
            onClick={() => setPage(label)}
            key={label}
          >
            <Icon size={18} />
            {label}
          </button>
        ))}

        <div className="sidebottom">
          <ShieldCheck size={18} />

          <span>
            <b>Privacy first</b>
            <small>Demo only. Avoid patient identifiers.</small>
          </span>
        </div>
      </aside>

      <main>
        <header>
          <button
            className="mobile"
            aria-label="Toggle navigation"
            onClick={() =>
              document.querySelector('.side')?.classList.toggle('open')
            }
          >
            <Menu />
          </button>

          <span>
            Workspace / <b>{page}</b>
          </span>

          <div className="pill">
            <i className={health === 'Connected' ? 'green' : ''} />
            {health}
          </div>
        </header>

        <div className="content">
          {/* OVERVIEW PAGE */}
          {page === 'Overview' && (
            <>
              <section className="hero">
                <div className="eyebrow">
                  <Sparkles size={14} />
                  MEDICAL KNOWLEDGE INTELLIGENCE
                </div>

                <h1>
                  Medical insights,
                  <br />
                  <span>connected by knowledge.</span>
                </h1>

                <p>
                  Explore extracted medical entities, inspect your knowledge
                  graph, and test retrieval-augmented answers.
                </p>

                <button
                  className="primary"
                  onClick={() => setPage('Medical AI Chat')}
                >
                  <MessageCircle size={17} />
                  Try Medical AI Chat
                </button>

                <div className="heroicon">
                  <HeartPulse size={76} />
                  <span />
                  <span />
                </div>
              </section>

              <div className="sectiontitle">
                <div>
                  <h2>Pipeline at a glance</h2>
                  <p>Current reported extraction results</p>
                </div>

                <span className="tag">
                  <Activity size={14} />
                  Project metrics
                </span>
              </div>

              <div className="metrics">
                {demoMetrics.map(([value, label, description], index) => {
                  const Icon = [
                    FileSearch,
                    Search,
                    CheckCircle2,
                    Network
                  ][index];

                  return (
                    <article className="metric" key={label}>
                      <span className={`mi m${index}`}>
                        <Icon size={19} />
                      </span>

                      <small>{label}</small>
                      <strong>{value}</strong>
                      <p>{description}</p>
                    </article>
                  );
                })}
              </div>

              <div className="twocol">
                <section className="panel">
                  <h2>Knowledge pipeline</h2>
                  <p>From medical text to contextual answers</p>

                  {[
                    ['01', 'Medical text chunks', 'Input data'],
                    ['02', 'SciSpaCy + UMLS', 'Entity extraction and linking'],
                    ['03', 'Validation + ontology', 'Filter and classify concepts'],
                    ['04', 'Medical knowledge graph', 'Nodes and relationships'],
                    ['05', 'Graph retrieval + LLM', 'Context-aware response']
                  ].map(([number, title, description]) => (
                    <div className="step" key={number}>
                      <span>{number}</span>

                      <div>
                        <b>{title}</b>
                        <small>{description}</small>
                      </div>
                    </div>
                  ))}
                </section>

                <section className="panel">
                  <h2>System status</h2>
                  <p>Local service connection</p>

                  <div className="service">
                    <Database />

                    <div>
                      <b>FastAPI backend</b>
                      <small>127.0.0.1:8000</small>
                    </div>

                    <strong>{health}</strong>
                  </div>

                  <div className="service">
                    <Bot />

                    <div>
                      <b>Graph + Ollama</b>
                      <small>Requires adapter verification</small>
                    </div>

                    <strong>Verify</strong>
                  </div>

                  <div className="note">
                    Metrics are extraction counts, not medical answer accuracy.
                  </div>
                </section>
              </div>
            </>
          )}

          {/* MEDICAL AI CHAT PAGE */}
          {page === 'Medical AI Chat' && (
            <>
              <div className="title">
                <h1>Medical AI Chat</h1>

                <p>
                  Test your graph retrieval and language-model workflow.
                </p>
              </div>

              <section className="panel chat">
                <div className="chathead">
                  <HeartPulse />

                  <div>
                    <b>MedGraph Assistant</b>
                    <small>Context-aware demo · Not for diagnosis</small>
                  </div>
                </div>

                <div className="msgs">
                  {messages.map((message, index) => (
                    <div
                      className={`msg ${message.role}`}
                      key={`${message.role}-${index}`}
                    >
                      <b>
                        {message.role === 'user' ? 'You' : 'MedGraph AI'}
                      </b>

                      <p>{message.text}</p>

                      {/* NEW: Generated image appears below answer text. */}
                      {message.role === 'assistant' && (
                        <MessageImages images={message.images} />
                      )}

                      {message.warning && (
                        <small className="warn">
                          {message.warning}
                        </small>
                      )}

                      {message.role === 'assistant' && (
                        <>
                          {message.model && (
                            <small className="disclaimer">
                              Model: {message.model}
                              {message.retrievedCount > 0
                                ? ` · Passages retrieved: ${message.retrievedCount}`
                                : ''}
                            </small>
                          )}

                          <MessageSources sources={message.sources} />

                          {message.concepts?.length > 0 && (
                            <div className="source-concepts">
                              <strong>Retrieved concepts:</strong>{' '}
                              {message.concepts.join(', ')}
                            </div>
                          )}

                          {message.text !== WELCOME_MESSAGE &&
                            (!message.sources ||
                              message.sources.length === 0) && (
                              <small className="warn">
                                No source passages were found in the frontend
                                response.
                                {message.responseKeys?.length
                                  ? ` Response fields: ${message.responseKeys.join(', ')}.`
                                  : ' Check the /api/ask response and Vite proxy.'}
                              </small>
                            )}
                        </>
                      )}
                    </div>
                  ))}

                  {busy && (
                    <p>Retrieving context and generating answer…</p>
                  )}
                </div>

                <div className="samples">
                  {[
                    'What are the common symptoms of dengue fever?',
                    'What are the warning signs of severe dengue?',
                    'How long does the febrile phase of dengue last?'
                  ].map(sample => (
                    <button
                      onClick={() => ask(sample)}
                      disabled={busy}
                      key={sample}
                    >
                      {sample}
                    </button>
                  ))}
                </div>

                <form
                  onSubmit={event => {
                    event.preventDefault();
                    ask();
                  }}
                >
                  <input
                    value={q}
                    onChange={event => setQ(event.target.value)}
                    placeholder="Ask a medical question…"
                    aria-label="Ask a medical question"
                    disabled={busy}
                  />

                  <button
                    className="primary"
                    type="submit"
                    disabled={busy || !q.trim()}
                    aria-label="Send question"
                  >
                    <Send size={17} />
                  </button>
                </form>

                <small className="disclaimer">
                  For project demonstration only. Not medical advice.
                </small>
              </section>
            </>
          )}

          {/* ENTITY EXPLORER PAGE */}
          {page === 'Entity Explorer' && (
            <>
              <div className="title">
                <h1>Entity Explorer</h1>
                <p>Search medical entities loaded from the Python API.</p>
              </div>

              <div className="search">
                <Search />

                <input
                  value={term}
                  onChange={event => setTerm(event.target.value)}
                  placeholder="Search medical entities…"
                  aria-label="Search medical entities"
                />
              </div>

              <div className="entities">
                {filteredEntities.map((entity, index) => (
                  <article className="entity" key={index}>
                    <Network />

                    <div>
                      <b>{name(entity)}</b>

                      <small>
                        {typeof entity === 'object'
                          ? entity.category ||
                            entity.type ||
                            entity.entity_type ||
                            'Medical entity'
                          : 'Medical entity'}
                      </small>
                    </div>
                  </article>
                ))}
              </div>

              {entities.length === 0 && (
                <div className="panel">
                  No entities loaded. Check that ontology_entities.json is
                  present and the API can read it.
                </div>
              )}

              {entities.length > 0 && filteredEntities.length === 0 && (
                <div className="panel">
                  No matching entities found. Try another search term.
                </div>
              )}
            </>
          )}

          {/* PIPELINE AND TOOLS PAGE */}
          {page === 'Pipeline & Tools' && (
            <>
              <div className="title">
                <h1>Pipeline &amp; Tools</h1>

                <p>
                  Current implementation versus planned enhancements.
                </p>
              </div>

              <div className="tools">
                {[
                  ['Python', 'Backend processing', 'Implemented'],
                  ['SciSpaCy', 'Biomedical entity extraction', 'Implemented'],
                  ['UMLS', 'Medical concept linking', 'Implemented'],
                  ['JSON graph', 'Entity and relationship files', 'Implemented'],
                  ['Graph retriever', 'Search graph context', 'Existing component'],
                  ['Ollama', 'Local language model', 'Verify integration'],
                  ['React + Vite', 'Frontend dashboard', 'Implemented'],
                  ['FastAPI', 'Frontend-to-backend API', 'Connected API'],
                  ['Sentence-BERT / BioBERT', 'Biomedical embeddings', 'Embedding model'],
                  ['FAISS', 'Vector similarity search', 'Vector retrieval']
                ].map(([title, description, status]) => (
                  <article className="tool" key={title}>
                    <span>
                      <GitBranch size={18} />
                    </span>

                    <h3>{title}</h3>
                    <p>{description}</p>
                    <small>{status}</small>
                  </article>
                ))}
              </div>
            </>
          )}

          <footer>
            <span>MedGraph AI · Local project workspace</span>
            <span>Demo only — not for clinical diagnosis</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

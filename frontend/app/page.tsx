"use client";

import { useState, useEffect, useRef } from "react";
import {
  getStatus,
  uploadCatalog,
  planCourses,
  clearIndex,
  type StudentProfile,
  type PlanResponse,
  type StatusResponse,
} from "@/lib/api";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
type Tab = "upload" | "ask" | "plan";

const TABS: { id: Tab; label: string }[] = [
  { id: "upload", label: "📁 Upload Catalog" },
  { id: "ask", label: "💬 Ask Questions" },
  { id: "plan", label: "📋 Plan My Term" },
];

const EMPTY_PROFILE: StudentProfile = {
  completed_courses: [],
  target_major: "",
  catalog_year: "",
  target_term: "",
  max_credits: null,
};

// ─────────────────────────────────────────────────────────────
// Helper components
// ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: StatusResponse | null }) {
  if (!status)
    return (
      <span className="text-xs text-gray-400 px-2 py-1 rounded bg-gray-100">
        Checking…
      </span>
    );
  return status.ready ? (
    <span className="text-xs text-green-700 px-2 py-1 rounded bg-green-100 font-medium">
      ✓ {status.chunk_count} chunks indexed
    </span>
  ) : (
    <span className="text-xs text-amber-700 px-2 py-1 rounded bg-amber-100 font-medium">
      ⚠ No catalog loaded
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-blue-600"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v8H4z"
      />
    </svg>
  );
}

function PlanOutput({ result }: { result: PlanResponse }) {
  const text = result.final_output || result.course_plan;

  // Convert structured plan sections to formatted HTML-like blocks
  const lines = text.split("\n");
  return (
    <div className="space-y-4">
      {/* Status banner */}
      {result.is_profile_complete ? (
        result.verified ? (
          <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-2">
            <span>✅</span>
            <span>Plan verified — all citations checked.</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2">
            <span>⚠️</span>
            <span>
              Plan generated but could not be fully verified. Some citations may
              be incomplete.
            </span>
          </div>
        )
      ) : (
        <div className="flex items-center gap-2 text-sm text-blue-700 bg-blue-50 border border-blue-200 rounded-lg px-4 py-2">
          <span>ℹ️</span>
          <span>Profile incomplete — please answer the clarifying questions below.</span>
        </div>
      )}

      {/* Clarifying questions */}
      {result.clarifying_questions.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h3 className="font-semibold text-yellow-800 mb-2">
            Clarifying Questions
          </h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-yellow-900">
            {result.clarifying_questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Main plan output */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 text-sm leading-relaxed whitespace-pre-wrap font-mono text-gray-800 max-h-[60vh] overflow-y-auto shadow-sm">
        {text}
      </div>

      {/* Citations */}
      {result.citations.length > 0 && (
        <details className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <summary className="cursor-pointer font-semibold text-sm text-gray-700">
            📚 Citations ({result.citations.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {result.citations.map((c, i) => (
              <li key={i} className="text-xs text-gray-600 font-mono">
                {c}
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* Assumptions */}
      {result.assumptions.length > 0 && (
        <details className="bg-orange-50 border border-orange-200 rounded-lg p-3">
          <summary className="cursor-pointer font-semibold text-sm text-orange-700">
            ⚠ Assumptions / Not in Catalog ({result.assumptions.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {result.assumptions.map((a, i) => (
              <li key={i} className="text-xs text-orange-800">
                • {a}
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* Failed citations */}
      {result.failed_citations.length > 0 && (
        <details className="bg-red-50 border border-red-200 rounded-lg p-3">
          <summary className="cursor-pointer font-semibold text-sm text-red-700">
            ❌ Failed Citations ({result.failed_citations.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {result.failed_citations.map((fc, i) => (
              <li key={i} className="text-xs text-red-800 font-mono">
                {fc}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Profile form (shared between Ask & Plan tabs)
// ─────────────────────────────────────────────────────────────

function ProfileForm({
  profile,
  onChange,
}: {
  profile: StudentProfile;
  onChange: (p: StudentProfile) => void;
}) {
  const set = (key: keyof StudentProfile, value: unknown) =>
    onChange({ ...profile, [key]: value });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Completed Courses{" "}
          <span className="text-gray-400 font-normal">(comma-separated)</span>
        </label>
        <input
          type="text"
          placeholder="e.g. CS101, MATH101, ENG101"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={profile.completed_courses.join(", ")}
          onChange={(e) =>
            set(
              "completed_courses",
              e.target.value
                .split(",")
                .map((s) => s.trim())
                .filter(Boolean)
            )
          }
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Target Major
        </label>
        <input
          type="text"
          placeholder="e.g. Computer Science"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={profile.target_major}
          onChange={(e) => set("target_major", e.target.value)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Catalog Year
        </label>
        <input
          type="text"
          placeholder="e.g. 2024-2025"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={profile.catalog_year}
          onChange={(e) => set("catalog_year", e.target.value)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Target Term
        </label>
        <select
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={profile.target_term}
          onChange={(e) =>
            set("target_term", e.target.value as "Fall" | "Spring" | "")
          }
        >
          <option value="">Select term…</option>
          <option value="Fall">Fall</option>
          <option value="Spring">Spring</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Max Credits
        </label>
        <input
          type="number"
          placeholder="e.g. 18"
          min={1}
          max={30}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={profile.max_credits ?? ""}
          onChange={(e) =>
            set(
              "max_credits",
              e.target.value ? parseInt(e.target.value, 10) : null
            )
          }
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Tab panels
// ─────────────────────────────────────────────────────────────

function UploadTab({
  onUploaded,
}: {
  onUploaded: () => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [clearExisting, setClearExisting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("Please select at least one PDF or HTML file.");
      return;
    }
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const res = await uploadCatalog(files, clearExisting);
      setMessage(res.message);
      setFiles([]);
      if (inputRef.current) inputRef.current.value = "";
      onUploaded();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!confirm("Clear all indexed catalog data? This cannot be undone.")) return;
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      await clearIndex();
      setMessage("Index cleared successfully.");
      onUploaded();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">📄 Upload your course catalog documents</p>
        <ul className="list-disc list-inside space-y-0.5 text-blue-700">
          <li>Accepted formats: PDF, HTML</li>
          <li>Multiple files supported (e.g. course catalog + syllabi)</li>
          <li>Documents are chunked and indexed into the vector store</li>
          <li>Clear existing index before uploading to start fresh</li>
        </ul>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Files
        </label>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.html,.htm"
          className="block w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
        {files.length > 0 && (
          <ul className="mt-2 text-sm text-gray-500 space-y-0.5">
            {files.map((f, i) => (
              <li key={i}>• {f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
            ))}
          </ul>
        )}
      </div>

      <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
        <input
          type="checkbox"
          checked={clearExisting}
          onChange={(e) => setClearExisting(e.target.checked)}
          className="rounded"
        />
        Clear existing index before uploading
      </label>

      <div className="flex gap-3">
        <button
          onClick={handleUpload}
          disabled={loading}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
        >
          {loading && <Spinner />}
          {loading ? "Processing…" : "Upload & Index"}
        </button>
        <button
          onClick={handleClear}
          disabled={loading}
          className="flex items-center gap-2 bg-red-50 hover:bg-red-100 disabled:opacity-50 text-red-700 border border-red-300 px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
        >
          Clear Index
        </button>
      </div>

      {message && (
        <div className="bg-green-50 border border-green-200 text-green-800 rounded-lg px-4 py-3 text-sm">
          ✅ {message}
        </div>
      )}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg px-4 py-3 text-sm">
          ❌ {error}
        </div>
      )}
    </div>
  );
}

function AskTab() {
  const [profile, setProfile] = useState<StudentProfile>({ ...EMPTY_PROFILE });
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PlanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please enter a question.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await planCourses({ user_query: query, student_profile: profile });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">💬 Ask a course-related question</p>
        <p className="text-blue-700">
          Fill in your profile, type your question, and the AI will search the
          catalog and answer with citations.
        </p>
      </div>

      <ProfileForm profile={profile} onChange={setProfile} />

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Your Question
        </label>
        <textarea
          rows={3}
          placeholder="e.g. What are the prerequisites for CS301?"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
      >
        {loading && <Spinner />}
        {loading ? "Thinking…" : "Ask"}
      </button>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg px-4 py-3 text-sm">
          ❌ {error}
        </div>
      )}
      {result && <PlanOutput result={result} />}
    </div>
  );
}

function PlanTab() {
  const [profile, setProfile] = useState<StudentProfile>({ ...EMPTY_PROFILE });
  const [query, setQuery] = useState(
    "Please create a full course plan for my next term."
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PlanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please describe what you need.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await planCourses({ user_query: query, student_profile: profile });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">📋 Generate a full term plan</p>
        <p className="text-blue-700">
          Fill in your academic profile and the AI will build a complete,
          cited course plan for your next term — checked by a verification agent.
        </p>
      </div>

      <ProfileForm profile={profile} onChange={setProfile} />

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Additional Instructions{" "}
          <span className="text-gray-400 font-normal">(optional)</span>
        </label>
        <textarea
          rows={2}
          placeholder="e.g. I want to focus on AI/ML courses this term."
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-green-300 text-white px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
      >
        {loading && <Spinner />}
        {loading
          ? "Generating plan… (this may take a few minutes)"
          : "Generate Term Plan"}
      </button>

      {loading && (
        <div className="text-sm text-gray-500 bg-gray-50 border border-gray-200 rounded-lg px-4 py-3">
          <p className="font-medium text-gray-700 mb-1">🤖 AI Agents Working…</p>
          <ol className="list-decimal list-inside space-y-1 text-gray-500">
            <li>Retriever Agent — searching catalog chunks…</li>
            <li>Planner Agent — building cited term plan…</li>
            <li>Verifier Agent — auditing citations…</li>
          </ol>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg px-4 py-3 text-sm">
          ❌ {error}
        </div>
      )}
      {result && <PlanOutput result={result} />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Root Page
// ─────────────────────────────────────────────────────────────

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [status, setStatus] = useState<StatusResponse | null>(null);

  const refreshStatus = async () => {
    try {
      const s = await getStatus();
      setStatus(s);
    } catch {
      setStatus({ chunk_count: 0, ready: false });
    }
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-700 to-blue-500 text-white shadow-lg">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold mb-1">
            📚 Prerequisite Course Planning Assistant
          </h1>
          <p className="text-blue-100 text-sm max-w-2xl">
            AI-powered academic advising using CrewAI agents, ChromaDB vector
            search, and Retrieval-Augmented Generation — every recommendation
            backed by catalog citations.
          </p>
          <div className="mt-3 flex items-center gap-3">
            <StatusBadge status={status} />
            <span className="text-xs text-blue-200">
              Powered by CrewAI + Ollama + ChromaDB
            </span>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6">
          <nav className="flex gap-0" aria-label="Tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-5 py-4 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? "border-blue-600 text-blue-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        {activeTab === "upload" && (
          <UploadTab onUploaded={refreshStatus} />
        )}
        {activeTab === "ask" && <AskTab />}
        {activeTab === "plan" && <PlanTab />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-4 text-center text-xs text-gray-400">
        Prerequisite Course Planning Assistant · CrewAI + LangChain + ChromaDB +
        Next.js
      </footer>
    </div>
  );
}

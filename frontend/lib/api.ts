/**
 * API client for the Course Planning Assistant FastAPI backend.
 */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface StudentProfile {
  completed_courses: string[];
  target_major: string;
  catalog_year: string;
  target_term: "Fall" | "Spring" | "";
  max_credits: number | null;
  grades?: Record<string, string>;
  transfer_credits?: string[];
}

export interface PlanRequest {
  user_query: string;
  student_profile: StudentProfile;
}

export interface PlanResponse {
  is_profile_complete: boolean;
  clarifying_questions: string[];
  missing_fields: string[];
  course_plan: string;
  final_output: string;
  citations: string[];
  assumptions: string[];
  verified: boolean;
  failed_citations: string[];
}

export interface StatusResponse {
  chunk_count: number;
  ready: boolean;
}

export interface UploadResponse {
  message: string;
  chunks_indexed: number;
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
  return res.json();
}

export async function uploadCatalog(
  files: File[],
  clearExisting = false
): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("clear_existing", clearExisting ? "true" : "false");

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function planCourses(req: PlanRequest): Promise<PlanResponse> {
  const res = await fetch(`${API_BASE}/api/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `Plan request failed: ${res.statusText}`);
  }
  return res.json();
}

export async function clearIndex(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/index`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `Clear failed: ${res.statusText}`);
  }
}

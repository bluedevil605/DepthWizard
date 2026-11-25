/**
 * API client for DepthWizard Backend
 */

import { upload } from '@vercel/blob/client';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');
const USE_BLOB_UPLOAD = import.meta.env.VITE_USE_BLOB_UPLOAD === 'true';

function apiUrl(path: string): string {
  if (import.meta.env.PROD && !API_BASE) {
    throw new Error('VITE_API_BASE_URL must be set for production builds.');
  }
  return `${API_BASE}${path}`;
}

function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), {
    ...init,
    credentials: API_BASE ? 'include' : 'same-origin',
  });
}

export async function fetchHealth(): Promise<any> {
  const res = await apiFetch('/api/health');
  if (!res.ok) throw new Error('Backend offline');
  return res.json();
}

export async function fetchSamples(): Promise<any> {
  const res = await apiFetch('/api/samples');
  if (!res.ok) throw new Error('Failed to load sample list');
  return res.json();
}

export async function processSample(sampleId: string, verticalExaggeration = 1.0, relativeExaggeration = 18.0, force = false): Promise<any> {
  const res = await apiFetch(
    `/api/process-sample/${sampleId}?vertical_exaggeration=${verticalExaggeration}&relative_exaggeration=${relativeExaggeration}&force=${force}`,
    { method: 'POST' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Processing error' }));
    throw new Error(err.detail || 'Failed to process sample');
  }
  return res.json();
}

export async function uploadImageAndProcess(
  file: File,
  referenceFile: File | null = null,
  verticalExaggeration = 1.0,
  relativeExaggeration = 18.0,
  force = false
): Promise<any> {
  if (USE_BLOB_UPLOAD) {
    const uploadToBlob = async (uploadFile: File) => {
      const safeName = uploadFile.name.replace(/[^A-Za-z0-9._-]/g, '_');
      return upload(`uploads/${crypto.randomUUID()}-${safeName}`, uploadFile, {
        access: 'public',
        handleUploadUrl: '/api/blob-upload',
      });
    };

    const [image, reference] = await Promise.all([
      uploadToBlob(file),
      referenceFile ? uploadToBlob(referenceFile) : Promise.resolve(null),
    ]);

    const res = await apiFetch('/api/process-blob', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input_pathname: image.pathname,
        input_filename: file.name,
        reference_pathname: reference?.pathname ?? null,
        reference_filename: referenceFile?.name ?? null,
        vertical_exaggeration: verticalExaggeration,
        relative_exaggeration: relativeExaggeration,
        force,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `Server error ${res.status}: ${res.statusText}` }));
      throw new Error(err.detail || err.message || JSON.stringify(err));
    }

    const contentType = res.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error(`Server returned unexpected format (${contentType}). Expected JSON.`);
    }
    return res.json();
  }

  const formData = new FormData();
  formData.append('file', file);
  if (referenceFile) {
    formData.append('reference_file', referenceFile);
  }
  formData.append('vertical_exaggeration', verticalExaggeration.toString());
  formData.append('relative_exaggeration', relativeExaggeration.toString());
  formData.append('force', force ? 'true' : 'false');

  const res = await apiFetch('/api/upload', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Server error ${res.status}: ${res.statusText}` }));
    const errorDetail = err.detail || err.message || JSON.stringify(err);
    throw new Error(errorDetail);
  }

  const contentType = res.headers.get('content-type');
  if (!contentType || !contentType.includes('application/json')) {
    throw new Error(`Server returned unexpected format (${contentType}). Expected JSON.`);
  }

  return res.json();
}

export async function fetchMeshMeta(metaUrl: string): Promise<any> {
  if (!metaUrl) return null;
  const res = await fetch(metaUrl);
  if (!res.ok) return null;
  const contentType = res.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return res.json();
  }
  return null;
}

export async function fetchValidationMetrics(): Promise<any> {
  const res = await apiFetch('/api/validation/metrics');
  if (!res.ok) throw new Error('Failed to fetch validation metrics');
  return res.json();
}

export async function fetchValidationBundle(sampleId = 'mount_st_helens'): Promise<any> {
  const res = await apiFetch(`/api/validation/bundle/${sampleId}`);
  if (!res.ok) throw new Error(`Failed to fetch validation bundle for ${sampleId}`);
  return res.json();
}

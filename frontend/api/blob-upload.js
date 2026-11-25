import { handleUpload } from '@vercel/blob/client';

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const allowedExtensions = new Set(['.png', '.jpg', '.jpeg', '.tif', '.tiff']);

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    response.setHeader('Allow', 'POST');
    return response.status(405).json({ error: 'Method not allowed' });
  }
  if (process.env.BLOB_UPLOADS_ENABLED !== 'true') {
    return response.status(503).json({ error: 'Blob uploads are not enabled for this deployment.' });
  }

  try {
    const result = await handleUpload({
      body: request.body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        const extension = pathname.slice(pathname.lastIndexOf('.')).toLowerCase();
        if (!pathname.startsWith('uploads/') || !allowedExtensions.has(extension)) {
          throw new Error('Only PNG, JPG, and TIFF uploads are supported.');
        }

        return {
          allowedContentTypes: [
            'image/png',
            'image/jpeg',
            'image/tiff',
            'application/octet-stream',
          ],
          maximumSizeInBytes: MAX_UPLOAD_BYTES,
          addRandomSuffix: true,
        };
      },
    });

    return response.status(200).json(result);
  } catch (error) {
    console.error('Blob upload authorization failed:', error);
    return response.status(400).json({ error: 'Blob upload could not be authorized.' });
  }
}

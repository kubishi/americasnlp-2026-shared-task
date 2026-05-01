// Only serve from americasnlp.proj.kubishi.com. Any other hostname
// (americasnlp-2026.pages.dev, preview URLs, etc.) gets a 404 so content
// never appears outside the canonical domain.

const ALLOWED_HOST = "americasnlp.proj.kubishi.com";

export const onRequest = async ({ request, next }) => {
  const url = new URL(request.url);
  if (url.hostname !== ALLOWED_HOST) {
    return new Response("Not Found", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  return next();
};

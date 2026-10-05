/** Agent docstrings are Markdown and include a PEAS block that the cards below already show:
 * strip the emphasis markers and stop before "PEAS". */
export function cleanDocstring(doc: string): string {
  const text = doc
    .replace(/\*\*|\*/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const cut = text.search(/\bPEAS\b/);
  const body = cut > 0 ? text.slice(0, cut).trim() : text;
  return body.replace(/[\s:;,-]+$/, "");
}

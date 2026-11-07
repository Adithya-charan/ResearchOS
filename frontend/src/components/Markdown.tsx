import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";

/** Markdown from the server is untrusted (it can quote document text). It is parsed to React elements (no raw HTML), then
 *  sanitised, and links get rel=noopener. `javascript:` URLs are removed by the sanitiser's default protocol allow-list. */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-doc text-sm text-ink">
      <ReactMarkdown rehypePlugins={[rehypeSanitize]} components={{ a: ({ href, children: c }) => <a href={href} rel="noopener noreferrer nofollow" target="_blank">{c}</a> }}>
        {children}
      </ReactMarkdown>
    </div>
  );
}

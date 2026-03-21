#!/usr/bin/env node
/**
 * Web search helper for plan-debate subagents.
 *
 * Uses multiple strategies in order of reliability:
 *   1. GitHub search (for code, repos, issues — via gh CLI)
 *   2. Fetch + extract text from URLs (for documentation pages)
 *   3. DuckDuckGo HTML search (for general web queries)
 *
 * Usage:
 *   node web-search.mjs search "langgraph agent tools"
 *   node web-search.mjs search "crewai multi-agent" --max 10
 *   node web-search.mjs fetch "https://docs.example.com/page"
 *   node web-search.mjs gh-search-code "class Agent" --repo langchain-ai/langgraph
 *   node web-search.mjs gh-search-repos "agentic framework"
 *   node web-search.mjs gh-readme langchain-ai/langgraph
 */

const args = process.argv.slice(2);
const command = args[0];

if (!command) {
	console.log(`Usage:
  node web-search.mjs search <query> [--max N]
  node web-search.mjs fetch <url>
  node web-search.mjs gh-search-code <query> --repo <owner/repo> [--max N]
  node web-search.mjs gh-search-repos <query> [--max N]
  node web-search.mjs gh-readme <owner/repo>
  node web-search.mjs gh-file <owner/repo> <path>`);
	process.exit(0);
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function getArg(name) {
	const idx = args.indexOf(name);
	return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

function getMax() {
	return parseInt(getArg("--max") || "10", 10);
}

/** Strip HTML tags and decode entities */
function stripHtml(html) {
	return html
		.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
		.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
		.replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
		.replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
		.replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
		.replace(/<[^>]+>/g, " ")
		.replace(/&nbsp;/g, " ")
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")
		.replace(/\s+/g, " ")
		.trim();
}

/** Extract meaningful text from HTML, keeping some structure */
function extractText(html, maxLength = 15000) {
	// Try to find main content area
	const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i)
		|| html.match(/<article[^>]*>([\s\S]*?)<\/article>/i)
		|| html.match(/<div[^>]*class="[^"]*content[^"]*"[^>]*>([\s\S]*?)<\/div>/i)
		|| html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);

	const content = mainMatch ? mainMatch[1] : html;

	// Convert some tags to text markers for structure
	let text = content
		.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, "\n\n## $2\n\n")
		.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, "\n- $1")
		.replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, "\n```\n$1\n```\n")
		.replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, "`$1`")
		.replace(/<br\s*\/?>/gi, "\n")
		.replace(/<p[^>]*>/gi, "\n\n");

	text = stripHtml(text);

	// Collapse multiple newlines
	text = text.replace(/\n{3,}/g, "\n\n").trim();

	if (text.length > maxLength) {
		text = text.slice(0, maxLength) + "\n\n[TRUNCATED]";
	}

	return text;
}

/** Run a shell command and return stdout */
async function exec(cmd) {
	const { execSync } = await import("node:child_process");
	try {
		return execSync(cmd, { encoding: "utf-8", timeout: 30000, stdio: ["pipe", "pipe", "pipe"] }).trim();
	} catch {
		return "";
	}
}

// ─── Commands ─────────────────────────────────────────────────────────────

async function searchWeb(query) {
	const max = getMax();
	const results = [];

	// Strategy 1: DuckDuckGo HTML lite
	try {
		const params = new URLSearchParams({ q: query, kl: "us-en" });
		const resp = await fetch(`https://lite.duckduckgo.com/lite/?${params}`, {
			headers: {
				"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
			},
			signal: AbortSignal.timeout(10000),
		});
		const html = await resp.text();

		// DDG lite puts results in a table with specific structure
		// Extract links from the results
		const linkRegex = /href="(https?:\/\/[^"]+)"[^>]*class="[^"]*result-link/gi;
		let match;
		while ((match = linkRegex.exec(html)) !== null && results.length < max) {
			results.push({ url: decodeURIComponent(match[1]), title: "", snippet: "", source: "duckduckgo" });
		}

		// Also try extracting from any external links
		if (results.length === 0) {
			const anyLinkRegex = /href="(https?:\/\/(?!duckduckgo\.com)[^"]+)"/gi;
			const seen = new Set();
			while ((match = anyLinkRegex.exec(html)) !== null && results.length < max) {
				const url = decodeURIComponent(match[1]);
				if (!seen.has(url) && !url.includes("duckduckgo.com")) {
					seen.add(url);
					results.push({ url, title: "", snippet: "", source: "duckduckgo" });
				}
			}
		}
	} catch { /* DDG failed, continue */ }

	// Strategy 2: If DDG failed, try GitHub search for the query
	if (results.length === 0) {
		try {
			const ghResult = await exec(`gh search repos "${query.replace(/"/g, '\\"')}" --json name,url,description --limit ${max}`);
			if (ghResult) {
				const repos = JSON.parse(ghResult);
				for (const repo of repos) {
					results.push({
						url: repo.url,
						title: repo.name,
						snippet: repo.description || "",
						source: "github-repos",
					});
				}
			}
		} catch { /* ignore */ }
	}

	console.log(JSON.stringify({ query, resultCount: results.length, results }, null, 2));
}

async function fetchUrl(url) {
	try {
		const resp = await fetch(url, {
			headers: {
				"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
				"Accept": "text/html,application/xhtml+xml,text/plain,application/json",
			},
			signal: AbortSignal.timeout(15000),
			redirect: "follow",
		});

		if (!resp.ok) {
			console.log(JSON.stringify({ url, error: `HTTP ${resp.status}`, content: "" }));
			return;
		}

		const contentType = resp.headers.get("content-type") || "";
		const body = await resp.text();

		let content;
		if (contentType.includes("json")) {
			// JSON — pretty print
			try {
				content = JSON.stringify(JSON.parse(body), null, 2).slice(0, 15000);
			} catch {
				content = body.slice(0, 15000);
			}
		} else if (contentType.includes("html")) {
			content = extractText(body);
		} else {
			// Plain text or other
			content = body.slice(0, 15000);
		}

		console.log(JSON.stringify({ url, contentType, contentLength: body.length, content }));
	} catch (e) {
		console.log(JSON.stringify({ url, error: e.message, content: "" }));
	}
}

async function ghSearchCode(query) {
	const repo = getArg("--repo");
	const max = getMax();
	if (!repo) {
		console.log(JSON.stringify({ error: "Missing --repo argument" }));
		return;
	}
	// Use spawn to avoid shell quoting issues
	const { execFileSync } = await import("node:child_process");
	try {
		const result = execFileSync("gh", [
			"search", "code", query, "--repo", repo,
			"--json", "path,textMatches", "--limit", String(max),
		], { encoding: "utf-8", timeout: 30000 });
		console.log(result.trim() || JSON.stringify({ query, repo, results: [] }));
	} catch {
		console.log(JSON.stringify({ query, repo, results: [] }));
	}
}

async function ghSearchRepos(query) {
	const max = getMax();
	const { execFileSync } = await import("node:child_process");
	try {
		const result = execFileSync("gh", [
			"search", "repos", query,
			"--json", "name,url,description,stargazersCount", "--limit", String(max),
		], { encoding: "utf-8", timeout: 30000 });
		console.log(result.trim() || JSON.stringify({ query, results: [] }));
	} catch {
		console.log(JSON.stringify({ query, results: [] }));
	}
}

async function ghReadme(repo) {
	// Fetch README via API, decode from base64
	const result = await exec(
		`gh api repos/${repo}/readme --jq '.content' 2>/dev/null | base64 -d`,
	);
	if (result) {
		const truncated = result.length > 15000 ? result.slice(0, 15000) + "\n\n[TRUNCATED]" : result;
		console.log(JSON.stringify({ repo, content: truncated }));
	} else {
		console.log(JSON.stringify({ repo, error: "Could not fetch README", content: "" }));
	}
}

async function ghFile(repo, filePath) {
	const result = await exec(
		`gh api repos/${repo}/contents/${filePath} --jq '.content' 2>/dev/null | base64 -d`,
	);
	if (result) {
		const truncated = result.length > 15000 ? result.slice(0, 15000) + "\n\n[TRUNCATED]" : result;
		console.log(JSON.stringify({ repo, path: filePath, content: truncated }));
	} else {
		console.log(JSON.stringify({ repo, path: filePath, error: "Could not fetch file", content: "" }));
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────

// Extract query: skip flags and their values
const queryParts = [];
for (let i = 1; i < args.length; i++) {
	if (args[i].startsWith("--")) {
		i++; // skip flag value too
	} else {
		queryParts.push(args[i]);
	}
}
const query = queryParts.join(" ");

switch (command) {
	case "search":
		await searchWeb(query);
		break;
	case "fetch":
		await fetchUrl(args[1]);
		break;
	case "gh-search-code":
		await ghSearchCode(query);
		break;
	case "gh-search-repos":
		await ghSearchRepos(query);
		break;
	case "gh-readme":
		await ghReadme(args[1]);
		break;
	case "gh-file":
		await ghFile(args[1], args[2]);
		break;
	default:
		console.log(`Unknown command: ${command}`);
		process.exit(1);
}

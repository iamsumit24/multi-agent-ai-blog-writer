import os
import sys
from pathlib import Path
import datetime

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import agent_tool

# env config
load_dotenv()

MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite")

# Sub-Agent: Planner
blog_planner = Agent(
   name="BlogPlanner",
   model=MODEL,
   description="Creates a practical, skimmable outline in Markdown.",
   instruction="""
You are a technical content strategist. Produce a clear Markdown outline with:
- Title
- Short intro
- 4–6 main sections (each with 2–3 bullets)
- Conclusion

If `codebase_context` exists in state, weave in specific sections/snippets.
Return only the outline in Markdown.
""",
   output_key="blog_outline",
)

class OutlineValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="OutlineValidationChecker",
           model=MODEL,
           description="Validates that the outline is usable.",
           instruction="""
Check the outline in state `blog_outline`. If it has a title, intro, 4–6 sections,                                                                                                                                                                                                                                                                                     and a conclusion, respond exactly "ok".
Otherwise respond exactly "retry" and list missing pieces.
""",
           output_key="validation_result",
       )

robust_blog_planner = LoopAgent(
   name="RobustBlogPlanner",
   description="Retries planning if validation fails.",
   sub_agents=[blog_planner, OutlineValidationChecker()],
   max_iterations=3,
)

# Sub-Agent: Writer
blog_writer = Agent(
   name="BlogWriter",
   model=MODEL,
   description="Writes a technical blog post from the outline.",
   instruction="""
You are a technical blog writer.

Use the outline stored in state key `blog_outline`.

Write a complete, detailed technical blog post in Markdown.

Requirements:
- Start with a strong H1 title.
- Add a short but engaging introduction.
- Expand every outline section into proper paragraphs.
- Use H2 and H3 headings.
- Add practical examples wherever useful.
- Add code snippets if relevant.
- Explain concepts clearly for developers.
- End with a strong conclusion.

Length requirement:
- Minimum 1200 words.
- Do not return only a summary.
- Do not return only titles or hooks.

Save the final blog post to state key `blog_post`.
Return only the complete blog post.
""",
   output_key="blog_post",
)

class BlogPostValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="BlogPostValidationChecker",
           model=MODEL,
           description="Validates the final post.",
           instruction="""
Check `blog_post` for: intro, clear sections matching the outline, conclusion, and                                                                                                                                                                                                                                                                                     technical clarity.
If passes, respond "ok". Else respond "retry" with the specific fixes.
""",
           output_key="validation_result",
       )

robust_blog_writer = LoopAgent(
   name="RobustBlogWriter",
   description="Retries writing if validation fails.",
   sub_agents=[blog_writer, BlogPostValidationChecker()],
   max_iterations=3,
)

# Expose planner/writer as tools so the root agent can call them explicitly
planner_tool = agent_tool.AgentTool(agent=robust_blog_planner)
writer_tool  = agent_tool.AgentTool(agent=robust_blog_writer)

# Root Agent: Plan → Write
root_agent = Agent(
   name="Blogger",
   model=MODEL,
   description="Minimal multi-agent blogger that plans and writes.",
   instruction=f"""
If the user gives a blog topic:

1. Call the planner tool to create a detailed outline.
2. Call the writer tool to create a full blog post from that outline.
3. Return the complete final blog post to the user.
4. After the blog post, add:
   - 3 alternate titles
   - 2 tweet-length hooks

Important:
- Do not summarize the blog post.
- Do not skip the blog post.
- The final answer must include the full article.
- The blog should look like a real blog post with:
  - H1 title
  - short introduction
  - multiple H2 sections
  - practical examples
  - conclusion
- Target length: 1200–1800 words.
""",
   tools=[
       planner_tool, # calls RobustBlogPlanner
       writer_tool,  # calls RobustBlogWriter
   ],
)

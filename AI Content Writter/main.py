import chainlit as cl
from agents import Agent, Runner, SQLiteSession, function_tool, RunHooks, RunContextWrapper
from openai.types.responses import ResponseTextDeltaEvent
from model_config import model_config
import fitz  # for pdf -> install PyMuPDF
from dotenv import load_dotenv
from ddgs import DDGS
load_dotenv()




session = SQLiteSession("blog_agent", "conversation.db")

config = model_config()



@function_tool
def web_search(query: str) -> str:
    """Fetch latest info from DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results)
    except Exception as e:
        return f"❌ Web search failed: {e}"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file (client product guide)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"❌ PDF extraction failed: {e}"



blog_agent = Agent(
    name="DraftingAgent",
    instructions="You are a content drafting agent. Based on the user's topic, use the web_search tool if needed to gather the latest information and research. Then, create a comprehensive first draft blog post with a clear structure: introduction, main body sections, and conclusion. Ensure it's informative and well-organized.",
    tools=[web_search]
)



seo_agent = Agent(
    name="SEOAgent📝✨",
    instructions="""
    You are an SEO Blog Optimizer Agent.
    Your responsibilities:
    - Take the provided blog draft and polish it.
    - Insert relevant trending keywords naturally (no keyword stuffing).
    - Use proper Markdown formatting:
      # for H1 (main title), ## for H2, ### for H3
    - Improve flow, readability, and engagement.
    - Keep the tone human, professional, and mobile-friendly.
    
    Important:
    - Only return the optimized blog content.
    - Do NOT generate meta description, hashtags, or alt text unless explicitly requested.
    """,
    tools=[web_search]
)



polish_agent = Agent(
    name="PolishAgent",
    instructions="You are a rewriting agent. Take the draft blog as input. Polish it for grammar, clarity, flow, and engagement. Improve the tone to be professional yet conversational, add engaging hooks, transitions, and calls-to-action. Make it fluent, error-free, and reader-friendly without changing the core meaning. For a 100-line blog, ensure the content remains concise while enhancing readability and engagement."
)

summarize_agent = Agent(
    name="SummarizeAgent",
    instructions="You're a summarization agent. Summarize the provided text comprehensively and accurately, capturing all details and main points in a highly structured and stylish format. Use bullet points with sub-bullets where necessary for clarity, ensuring no data is missed. Highlight important details using **bold text** and emphasize critical information with *italics*. Present the summary in a visually appealing layout, prioritizing readability and completeness. Respond in Urdu if the user prefers it. 😊",
)

main_agent = Agent(
    name="MainAgent",
    instructions="""You are a helpful AI assistant.
    - Talk to the user in warm and friendly tone.
    - Use relevant emojis with response""",
    handoffs=[blog_agent, summarize_agent],
    tools=[web_search]
)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="📝 Draft a Blog", message="Create a draft blog on the topic: AI in Marketing"),
        cl.Starter(label="🔍 SEO Optimize", message="SEO optimize this blog draft for better Google ranking."),
        cl.Starter(label="✨ Polish Content", message="Rewrite and polish the blog to make it more professional."),
        cl.Starter(label="📄 Import PDF Info", message="Pull key details from client's product guide PDF."),
        cl.Starter(label="🎭 Change Tone", message="Rewrite this blog in a casual tone."),
        cl.Starter(label="📤 Export Content", message="Export my blog as PDF and DOCX."),
    ]

@cl.on_message
async def handle_message(message: cl.Message):
    # Initialize a main message for non-blog responses
    msg = cl.Message(content="🤔 Thinking...")
    await msg.send()
    
    # 🗂️ Handle file uploads (summarization)
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_path = element.path
                if not file_path:
                    await cl.Message(content="⚠️ File path not found.").send()
                    continue
                lower = file_path.lower()
                text = None
                try:
                    if lower.endswith(".pdf"):
                        text = extract_text_from_pdf(file_path)
                except Exception as e:
                    await cl.Message(content=f"❌ Error extracting text: {e}").send()
                    continue
                if not text:
                    await cl.Message(content="⚠️ No text extracted.").send()
                    continue
                try:
                    result = Runner.run_streamed(
                        summarize_agent,
                        input=text,
                        run_config=config,
                        session=session,
                        
                    )
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            await msg.stream_token(event.data.delta)
                    msg.content = result.final_output
                    await msg.update()
                except Exception as e:
                    await cl.Message(content=f"❌ Summarization failed: {e}").send()
        return
    
    user_input_lower = message.content.lower()

    # Check user intent for blog generation
    if "blog" in user_input_lower or "article" in user_input_lower or "content" in user_input_lower:
        # Blog pipeline: Silent draft/polish, show progress and final SEO
        try:
            msg.content =  "📊✍️ Crafting your SEO-optimized blog with powerful insights… ⏳ please wait! 🚀"
            await msg.update()
            # Draft (silent, update progress message)
            draft_msg = cl.Message(content="📝 Generating draft...")
            await draft_msg.send()
            blog_response = await Runner.run(
                blog_agent,
                input=message.content,
                run_config=config,
                session=session,
               
            )
            if not blog_response.final_output:
                await cl.Message(content="❌ Drafting failed: No output generated.").send()
                return
            draft_msg.content = "📝 Draft Generated ✅"
            await draft_msg.update()
            
            # Polish (silent, update progress message)
            polish_msg = cl.Message(content="✨ Polishing the content...")
            await polish_msg.send()
            polish_response = await Runner.run(
                polish_agent,
                input=blog_response.final_output[:10000],
                run_config=config,
                session=session,
           
            )
            if not polish_response.final_output:
                await cl.Message(content="❌ Polishing failed: No output generated.").send()
                return
            polish_msg.content = "✨ Polished Content ✅"
            await polish_msg.update()
            
            # SEO (stream final output)
            seo_msg = cl.Message(content="📈 Optimizing for SEO...")
            await seo_msg.send()
            seo_response = Runner.run_streamed(
                seo_agent,
                input=polish_response.final_output,
                run_config=config,
                session=session,
               
            )
            async for event in seo_response.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    await seo_msg.stream_token(event.data.delta)
            seo_msg.content = seo_response.final_output
            await seo_msg.update()
            if not seo_response.final_output:
                await cl.Message(content="❌ SEO optimization failed: No output generated.").send()
                return

        except Exception as e:
            print(f"Blog pipeline error: {str(e)}")
            await cl.Message(content=f"❌ Error in blog pipeline: {str(e)}").send()
            return
    
    else:
        # Non-blog: Run MainAgent only
        try:
            main_response = Runner.run_streamed(
                main_agent,
                input=message.content,
                run_config=config,
                session=session,
             
            )
            async for event in main_response.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    await msg.stream_token(event.data.delta)
            msg.content = main_response.final_output
            await msg.update()
        except Exception as e:
            print(f"Main agent error: {str(e)}")
            await cl.Message(content=f"❌ Main agent failed: {str(e)}").send()
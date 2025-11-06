import openai
import os
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def summarize_text(text: str) -> str:
    """
    Summarize text using OpenAI GPT
    """
    try:
        # Truncate very long text to avoid token limits
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        prompt = f"""
        Please summarize the following LinkedIn post in one concise sentence (maximum 15 words). 
        Focus on the main point or key takeaway. Avoid hashtags and marketing fluff.
        
        LinkedIn Post:
        {text}
        
        Summary:
        """
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries of LinkedIn posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3,
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Clean up the summary
        summary = summary.replace('"', '').replace('Summary:', '').strip()
        
        logger.info(f"Generated summary: {summary}")
        return summary
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error generating summary. Please check your API key and try again."
        )
    except Exception as e:
        logger.error(f"Unexpected summarization error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error generating summary"
        )
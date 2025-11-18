"""
LLM client for CIVI-GENESIS using Google's Gemini API.
Handles citizen reaction generation and expert summaries.
Supports automatic API key rotation when quota is exceeded.
"""
import json
import re
import logging
import time
from typing import Dict, Any, Optional, List
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google's Gemini API with automatic key rotation."""
    
    def __init__(self, api_key: str, backup_keys: Optional[List[str]] = None):
        """
        Initialize the Gemini client with API key rotation support.
        
        Args:
            api_key: Primary Google API key for Gemini.
            backup_keys: Optional list of backup API keys to rotate through.
        """
        # Store all API keys
        self.api_keys = [api_key]
        if backup_keys:
            self.api_keys.extend(backup_keys)
        
        self.current_key_index = 0
        self.exhausted_keys = set()  # Track keys that hit daily quota
        self.all_keys_exhausted = False  # Flag when ALL keys are out of quota
        
        # Configure with primary key
        self._switch_api_key(0)
        
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 4.5  # 60s / 15 requests = 4s, add buffer
    
    def _switch_api_key(self, key_index: int):
        """Switch to a different API key."""
        if key_index >= len(self.api_keys):
            logger.error("No more API keys available!")
            return False
        
        self.current_key_index = key_index
        current_key = self.api_keys[key_index]
        
        # Reconfigure with new key
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Mask key for logging (show first 8 and last 4 chars)
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "***"
        logger.info(f"🔄 Switched to API key #{key_index + 1}/{len(self.api_keys)} ({masked_key})")
        return True
    
    def _rotate_to_next_key(self) -> bool:
        """
        Rotate to the next available API key.
        
        Returns:
            True if rotation successful, False if no keys available.
        """
        # Mark current key as exhausted
        self.exhausted_keys.add(self.current_key_index)
        
        # Try to find next available key
        for i in range(len(self.api_keys)):
            if i not in self.exhausted_keys:
                if self._switch_api_key(i):
                    logger.info(f"✅ Successfully rotated to backup key #{i + 1}")
                    return True
        
        # All keys exhausted
        self.all_keys_exhausted = True
        logger.error("❌ All API keys have exhausted their daily quota!")
        return False
    
    def _rate_limit(self):
        """Enforce rate limiting to stay within free tier quota (15 req/min)."""
        self.request_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s (request #{self.request_count})")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _call_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Gemini API with exponential backoff retry logic and automatic key rotation.
        
        Args:
            prompt: The prompt to send to the model.
            max_retries: Maximum number of retry attempts per key.
        
        Returns:
            Response text or None if all retries fail.
        """
        # Stop immediately if all keys are exhausted
        if self.all_keys_exhausted:
            return None
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    # Check if it's a daily quota limit (200 requests/day)
                    if "GenerateRequestsPerDayPerProjectPerModel" in error_msg or "limit: 200" in error_msg:
                        logger.warning("⚠️ Daily quota exceeded for current API key")
                        
                        # Try to rotate to next API key
                        if self._rotate_to_next_key():
                            logger.info("🔄 Continuing with backup key...")
                            # Reset attempt counter for new key
                            attempt = 0
                            continue
                        else:
                            # All keys exhausted - stop immediately
                            logger.error("❌ All API keys exhausted. Cannot continue.")
                            logger.error("   Solutions:")
                            logger.error("   1. Wait for quota reset (midnight UTC)")
                            logger.error("   2. Add more backup API keys to .env")
                            logger.error("   3. Use RULE_BASED or NN_ONLY mode")
                            logger.error("   4. Upgrade to paid tier")
                            logger.error("")
                            logger.error("⛔ Stopping further API calls to avoid waste.")
                            return None  # Stop immediately, don't retry
                    
                    # Handle per-minute rate limits
                    retry_match = re.search(r'retry in ([\d.]+)s', error_msg)
                    if retry_match:
                        retry_delay = float(retry_match.group(1)) + 1  # Add 1s buffer
                    else:
                        retry_delay = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    
                    # Only retry if delay is reasonable (< 2 minutes)
                    if retry_delay < 120 and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay:.1f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Rate limit exceeded with long delay ({retry_delay:.1f}s) or max retries reached")
                        return None
                else:
                    logger.error(f"Error calling Gemini API: {e}")
                    return None
        
        return None
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response text.
        
        Args:
            text: Raw text response from LLM.
        
        Returns:
            Parsed JSON dict or None if parsing fails.
        """
        # Try to find JSON object in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        
        # If no match found, try parsing the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to extract JSON from response: {text[:200]}")
            return None
    
    def generate_citizen_reaction(
        self,
        citizen_profile: Dict[str, Any],
        current_state: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a citizen's reaction to a policy using Gemini.
        
        Args:
            citizen_profile: Dict with citizen attributes (age, income_level, profession, etc.).
            current_state: Dict with current happiness, policy_support, income.
            policy: Dict with title, description, domain.
        
        Returns:
            Dict with new_happiness, new_policy_support, income_delta, short_reason, diary_entry.
        """
        system_prompt = """You are simulating how a fictional citizen in a small city reacts to a new policy. 
You receive citizen_profile, current_state, and policy. 
Respect citizen attributes such as income_level, profession, risk_tolerance, openness_to_change, 
political_view, and city_zone. Produce realistic but synthetic reactions.

Output ONLY valid JSON with these exact keys:
- new_happiness: float between 0 and 1
- new_policy_support: float between -1 and 1
- income_delta: float (change in income, can be negative, zero, or positive)
- short_reason: string, 1-2 sentences explaining the reaction
- diary_entry: string, 3-5 sentences in first-person perspective

Do not include any explanation outside the JSON."""
        
        user_prompt = f"""Citizen Profile:
{json.dumps(citizen_profile, indent=2)}

Current State:
{json.dumps(current_state, indent=2)}

Policy:
{json.dumps(policy, indent=2)}

Generate the citizen's reaction as JSON."""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            response_text = self._call_with_retry(full_prompt)
            if not response_text:
                raise Exception("Failed to get response after retries")
            
            result_json = self._extract_json(response_text)
            
            if result_json:
                # Validate and clamp values
                result_json["new_happiness"] = max(0.0, min(1.0, float(result_json.get("new_happiness", 0.5))))
                result_json["new_policy_support"] = max(-1.0, min(1.0, float(result_json.get("new_policy_support", 0.0))))
                result_json["income_delta"] = float(result_json.get("income_delta", 0.0))
                result_json["short_reason"] = str(result_json.get("short_reason", "No reason provided."))
                result_json["diary_entry"] = str(result_json.get("diary_entry", "No diary entry."))
                return result_json
            else:
                # Fallback with neutral values
                logger.warning("Failed to parse LLM response, using fallback values")
                return {
                    "new_happiness": current_state.get("happiness", 0.5),
                    "new_policy_support": 0.0,
                    "income_delta": 0.0,
                    "short_reason": "Unable to generate detailed reaction.",
                    "diary_entry": "Today was an ordinary day."
                }
        
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {
                "new_happiness": current_state.get("happiness", 0.5),
                "new_policy_support": 0.0,
                "income_delta": 0.0,
                "short_reason": "Error generating reaction.",
                "diary_entry": "System error occurred."
            }
    
    def generate_expert_summary(
        self,
        step_stats: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate expert perspectives on simulation results.
        
        Args:
            step_stats: Aggregated statistics for the current step.
            policy: Policy information.
        
        Returns:
            Dict with economist_view, activist_view, business_owner_view.
        """
        system_prompt = """You are summarizing the state of a synthetic society simulation given aggregated metrics and a policy description.
You must output three perspectives: an economist, a social activist, and a small business owner.
Use the metrics to ground your reasoning and highlight which groups are most impacted.

Output ONLY valid JSON with these exact keys:
- economist_view: string, a short paragraph (3-5 sentences)
- activist_view: string, a short paragraph (3-5 sentences)
- business_owner_view: string, a short paragraph (3-5 sentences)

Do not include any explanation outside the JSON."""
        
        user_prompt = f"""Policy:
{json.dumps(policy, indent=2)}

Current Step Statistics:
{json.dumps(step_stats, indent=2)}

Generate expert perspectives as JSON."""
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            response_text = self._call_with_retry(full_prompt)
            if not response_text:
                raise Exception("Failed to get response after retries")
            
            result_json = self._extract_json(response_text)
            
            if result_json:
                return {
                    "economist_view": str(result_json.get("economist_view", "No economist view available.")),
                    "activist_view": str(result_json.get("activist_view", "No activist view available.")),
                    "business_owner_view": str(result_json.get("business_owner_view", "No business owner view available."))
                }
            else:
                logger.warning("Failed to parse expert summary, using fallback")
                return {
                    "economist_view": "Analysis unavailable due to parsing error.",
                    "activist_view": "Analysis unavailable due to parsing error.",
                    "business_owner_view": "Analysis unavailable due to parsing error."
                }
        
        except Exception as e:
            logger.error(f"Error generating expert summary: {e}")
            
            # Generate data-driven fallback summaries instead of generic messages
            avg_happiness = step_stats.get('avg_happiness', 0.5)
            avg_support = step_stats.get('avg_support', 0.0)
            avg_income = step_stats.get('avg_income', 50000)
            policy_title = policy.get('title', 'policy')
            policy_domain = policy.get('domain', 'general')
            
            # Calculate sentiment indicators
            happiness_trend = "positive" if avg_happiness > 0.6 else "negative" if avg_happiness < 0.4 else "neutral"
            support_trend = "supportive" if avg_support > 0.2 else "opposed" if avg_support < -0.2 else "mixed"
            income_level = "high" if avg_income > 60000 else "low" if avg_income < 40000 else "moderate"
            
            return {
                "economist_view": f"The {policy_title} shows {happiness_trend} economic sentiment with average income at ${avg_income:,.0f}. Citizens are {support_trend} of this {policy_domain} policy. The {income_level} income levels suggest this policy's economic impact varies significantly across different income groups, requiring careful monitoring of distributional effects.",
                
                "activist_view": f"This {policy_domain} policy reveals concerning social dynamics. With {avg_happiness:.1%} average happiness and {support_trend} public opinion, we see clear evidence of unequal impact. The community needs stronger protections for vulnerable groups and more equitable policy design that addresses root causes of social inequality.",
                
                "business_owner_view": f"From a business perspective, the {policy_title} creates {happiness_trend} market conditions. With average customer income at ${avg_income:,.0f}, this impacts purchasing power and market demand. The {support_trend} public sentiment suggests businesses should prepare for regulatory changes and adapt operations accordingly."
            }


# Factory function for easier instantiation
def create_gemini_client(api_key: str, backup_keys: Optional[List[str]] = None) -> GeminiClient:
    """
    Create and return a GeminiClient instance with optional backup keys.
    
    Args:
        api_key: Primary API key.
        backup_keys: Optional list of backup API keys for automatic rotation.
    """
    return GeminiClient(api_key, backup_keys)

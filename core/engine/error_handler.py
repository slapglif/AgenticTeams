from typing import Dict, Any

class ErrorHandler:
    def format_error_message(self, error_info: Dict[str, Any]) -> str:
        """Format error information into a readable message."""
        try:
            # Extract key information
            explanation = error_info["analysis"]["explanation"]
            severity = error_info["analysis"]["severity"]
            impact = error_info["analysis"]["impact"]
            actions = error_info["analysis"]["recommended_actions"]
            recovery = error_info["recovery"]["immediate_actions"]
            
            # Format message sections
            sections = [
                "Error Explanation:",
                f"{explanation}\n",
                "SEVERITY:",
                f"{severity}\n",
                "IMPACT:",
                f"{impact}\n",
                "ACTION ITEMS:",
                "\n".join(f"- {action}" for action in actions),
                "\nRECOVERY STEPS:",
                "\n".join(f"- {step}" for step in recovery)
            ]
            
            return "\n".join(sections)
            
        except Exception as e:
            logger.error(f"Error formatting error message: {e}")
            return str(error_info) 
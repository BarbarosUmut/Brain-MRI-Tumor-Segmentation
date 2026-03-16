
import ollama

class OllamaClient:
    def __init__(self, model='llama3'):
        self.model = model
        
    def analyze_tumor(self, area, location_hex, classification="HGG/LGG"):
        """
        Generate an analysis report based on tumor metrics.
        """
        prompt = f"""
        You are a medical imaging assistant. 
        A brain tumor has been detected in an MRI scan with the following properties:
        - Estimated Area: {area} pixels (approximate)
        - Location (Centroid): {location_hex}
        
        Please provide a concise analysis summary. 
        Include:
        1. A note on the significance of the size (generic medical knowledge).
        2. Potential functional areas of the brain that might be affected given the location (assume standard orientation if not specified).
        3. A disclaimer that this is automated and requires a radiologist's review.
        
        Keep it professional and clinical.
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}. Ensure 'ollama serve' is running and model '{self.model}' is pulled."

if __name__ == "__main__":
    # Test
    client = OllamaClient()
    print(client.analyze_tumor(500, (120, 120)))

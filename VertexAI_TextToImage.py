import argparse
from vertexai.preview.vision_models import ImageGenerationModel

def generate_image(project_id, location, output_file, prompt):
  """Generiert ein Bild basierend auf einer Textbeschreibung.

  Args:
    project_id: Projekt-ID.
    location: Region
    output_file: Pfad des Ausgabebilds
    prompt: Text, der das zu generierende Bild beschreibt.
  """

  # Initialisiere den Vertex AI-Client
  vertexai.init(project=project_id, location=location)

  # Lade das Bildgenerierungsmodell
  model = ImageGenerationModel.from_pretrained("imagegenerationce")

  # Generiere ein Bild
  images = model.generate_images(prompt=prompt)

  # Speichere das erste generierte Bild
  images[0].save(location=output_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--project", help="Your project ID")
  parser.add_argument("--location", help="Your project location")
  parser.add_argument("--output_file", help="Path to save the generated image")
  parser.add_argument("--prompt", help="Text prompt to generate image")
  args = parser.parse_args()

  generate_image(args.project, args.location, args.output_file, args.prompt)

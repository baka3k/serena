from agno.models.google import Gemini
from agno.playground import Playground, serve_playground_app

from serena.agno import create_agno_agent

if __name__ == "__main__":
    serena_agent = create_agno_agent(
        project_file_path="myproject.yml",
        model=Gemini(id="gemini-2.5-pro-exp-03-25"),
    )
    app = Playground(agents=[serena_agent]).get_app()
    serve_playground_app(app, reload=False)

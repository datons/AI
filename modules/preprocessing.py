import requests
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from urllib.parse import urlparse


class BOCYLMarkdownExporter:
    def __init__(self, home_dir: str):
        self.home_dir = Path(home_dir)

    def parse_source(self, source: str) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        else:
            tree = ET.parse(source)
            root = tree.getroot()
        return self._extract_markdown(root)

    def _extract_markdown(self, root) -> str:
        markdown_lines = []

        titulo = root.find(".//titulo")
        if titulo is not None:
            markdown_lines.append(f"# {titulo.text.strip()}\n")

        metadata_fields = [
            "numeroEdicion",
            "fechaPublicacion",
            "fechaDisposicion",
            "seccion",
            "subseccion",
            "apartado",
            "organismo",
            "suborganismo",
            "rango",
            "numeroOficial",
            "entidadPublicadora",
        ]
        for field in metadata_fields:
            el = root.find(f".//{field}")
            if el is not None and el.text:
                markdown_lines.append(
                    f"**{field.replace('_', ' ').title()}:** {el.text.strip()}"
                )

        markdown_lines.append("\n---\n")

        texto = root.find(".//texto")
        if texto is not None:
            for p in texto.findall(".//p"):
                line = p.text.strip() if p.text else ""
                if not line:
                    continue
                if re.match(r"^Artículo\s+\d+[º.]", line):
                    markdown_lines.append(f"## {line}")
                elif line.isupper() and len(line.split()) <= 10:
                    markdown_lines.append(f"### {line}")
                else:
                    markdown_lines.append(line)

        return "\n\n".join(markdown_lines)

    def export(self, input_source: str, output_name: str = None):
        # Determine output filename
        if output_name is None:
            name = (
                Path(urlparse(input_source).path).stem
                if input_source.startswith("http")
                else Path(input_source).stem
            )
            output_name = name + ".md"

        # Prepare output path
        output_path = self.home_dir / output_name
        self.home_dir.mkdir(parents=True, exist_ok=True)

        # Check if the file already exists
        if output_path.exists():
            print(f"File {output_path} already exists. Skipping export.")
            return str(output_path)

        # Proceed with export if file doesn't exist
        markdown = self.parse_source(input_source)
        output_path.write_text(markdown, encoding="utf-8")
        return str(output_path)

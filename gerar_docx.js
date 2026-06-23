const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, TableOfContents, HeadingLevel, BorderStyle,
  WidthType, ShadingType, ExternalHyperlink, PageNumber, Header, Footer, PageBreak,
} = require("docx");

const SRC = "RELATORIO.md";
const OUT = "RELATORIO.docx";

const CONTENT_WIDTH = 9360; // US Letter, margens de 1"
const MONO = "Consolas";
const CODE_FILL = "F4F4F4";
const QUOTE_FILL = "FBF6E9";

// ---------- parsing inline (negrito, código, links) ----------
function parseInline(text, baseOpts = {}) {
  const runs = [];
  // regex que captura **bold**, `code`, [text](url)
  const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\[([^\]]+)\]\(([^)]+)\))/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      runs.push(new TextRun({ text: text.slice(last, m.index), ...baseOpts }));
    }
    if (m[1]) {
      runs.push(new TextRun({ text: m[2], bold: true, ...baseOpts }));
    } else if (m[3]) {
      runs.push(new TextRun({
        text: m[4], font: MONO, size: 19,
        shading: { type: ShadingType.CLEAR, fill: CODE_FILL },
        ...baseOpts,
      }));
    } else if (m[5]) {
      const label = m[6];
      const url = m[7];
      if (url.startsWith("#")) {
        // âncora interna -> apenas texto (sem link), mantém legibilidade
        runs.push(new TextRun({ text: label, ...baseOpts }));
      } else {
        runs.push(new ExternalHyperlink({
          link: url,
          children: [new TextRun({ text: label, style: "Hyperlink", ...baseOpts })],
        }));
      }
    }
    last = re.lastIndex;
  }
  if (last < text.length) {
    runs.push(new TextRun({ text: text.slice(last), ...baseOpts }));
  }
  if (runs.length === 0) runs.push(new TextRun({ text: "", ...baseOpts }));
  return runs;
}

// ---------- helpers de bloco ----------
function codeParagraph(line, indentLeft = 0) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: CODE_FILL },
    spacing: { before: 0, after: 0, line: 240 },
    indent: indentLeft ? { left: indentLeft } : undefined,
    children: [new TextRun({
      text: line.length ? line : " ",
      font: MONO, size: 18, color: "1A1A1A",
    })],
  });
}

function buildTable(rows) {
  // rows: array de arrays de células (strings). rows[0] é o cabeçalho.
  const nCols = rows[0].length;
  const colWidth = Math.floor(CONTENT_WIDTH / nCols);
  const colWidths = Array(nCols).fill(colWidth);
  colWidths[nCols - 1] = CONTENT_WIDTH - colWidth * (nCols - 1);

  const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const tableRows = rows.map((cells, rIdx) =>
    new TableRow({
      tableHeader: rIdx === 0,
      children: cells.map((cell, cIdx) =>
        new TableCell({
          borders,
          width: { size: colWidths[cIdx], type: WidthType.DXA },
          shading: rIdx === 0
            ? { type: ShadingType.CLEAR, fill: "2E5E8C" }
            : { type: ShadingType.CLEAR, fill: rIdx % 2 === 0 ? "F2F6FA" : "FFFFFF" },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          children: [new Paragraph({
            spacing: { before: 20, after: 20 },
            children: parseInline(cell, rIdx === 0 ? { bold: true, color: "FFFFFF" } : {}),
          })],
        })
      ),
    })
  );

  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows,
  });
}

function hr() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "9DB8D2", space: 1 } },
    children: [new TextRun("")],
  });
}

// ---------- parser principal de blocos ----------
function parseBlocks(lines, opts = {}) {
  const out = [];
  const quoteIndent = opts.quote ? 360 : 0;
  let i = 0;

  while (i < lines.length) {
    let line = lines[i];

    // linha em branco
    if (line.trim() === "") { i++; continue; }

    // regra horizontal
    if (/^---+\s*$/.test(line.trim())) { out.push(hr()); i++; continue; }

    // bloco de citação
    if (line.startsWith(">")) {
      const inner = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        inner.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      const innerEls = parseBlocks(inner, { ...opts, quote: true });
      out.push(...innerEls);
      continue;
    }

    // bloco de código
    if (line.trim().startsWith("```")) {
      i++;
      const code = [];
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++; // pula a cerca de fechamento
      code.forEach((c) => out.push(codeParagraph(c, quoteIndent)));
      out.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] }));
      continue;
    }

    // tabela
    if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|?[-: |]+\|?\s*$/.test(lines[i + 1])) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const rows = tableLines
        .filter((l, idx) => idx !== 1) // remove separador
        .map((l) =>
          l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim())
        );
      out.push(buildTable(rows));
      out.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] }));
      continue;
    }

    // títulos
    let mh;
    if ((mh = line.match(/^####\s+(.*)$/))) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: parseInline(mh[1]) }));
      i++; continue;
    }
    if ((mh = line.match(/^###\s+(.*)$/))) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: parseInline(mh[1]) }));
      i++; continue;
    }
    if ((mh = line.match(/^##\s+(.*)$/))) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: out.length > 0, children: parseInline(mh[1]) }));
      i++; continue;
    }
    if ((mh = line.match(/^#\s+(.*)$/))) {
      out.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: parseInline(mh[1]) }));
      i++; continue;
    }

    // lista com marcador
    if (/^\s*-\s+/.test(line)) {
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        const txt = lines[i].replace(/^\s*-\s+/, "");
        out.push(new Paragraph({
          numbering: { reference: "bullets", level: 0 },
          indent: quoteIndent ? { left: 720 + quoteIndent, hanging: 360 } : undefined,
          children: parseInline(txt),
        }));
        i++;
      }
      continue;
    }

    // lista numerada
    if (/^\s*\d+\.\s+/.test(line)) {
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const txt = lines[i].replace(/^\s*\d+\.\s+/, "");
        out.push(new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          indent: quoteIndent ? { left: 720 + quoteIndent, hanging: 360 } : undefined,
          children: parseInline(txt),
        }));
        i++;
      }
      continue;
    }

    // parágrafo normal
    out.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 120, line: 276 },
      indent: quoteIndent ? { left: quoteIndent } : undefined,
      shading: opts.quote ? { type: ShadingType.CLEAR, fill: QUOTE_FILL } : undefined,
      children: parseInline(line),
    }));
    i++;
  }

  return out;
}

// ---------- montagem do documento ----------
const md = fs.readFileSync(SRC, "utf8");
let lines = md.split("\n");

// Cabeçalho especial (título + subtítulo) e substituição do "Sumário" por TOC real
const elements = [];

// Título (primeira linha "# ")
let idx = 0;
while (idx < lines.length && lines[idx].trim() === "") idx++;
const titleMatch = lines[idx].match(/^#\s+(.*)$/);
if (titleMatch) {
  elements.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 2400, after: 240 },
    children: [new TextRun({ text: titleMatch[1], bold: true, size: 44, font: "Arial", color: "1F3864" })],
  }));
  idx++;
}
// Subtítulo (linhas "### " antes do primeiro "## ")
while (idx < lines.length && !/^##\s/.test(lines[idx])) {
  const sub = lines[idx].match(/^###\s+(.*)$/);
  if (sub) {
    elements.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: sub[1], size: 26, font: "Arial", color: "2E5E8C", italics: true })],
    }));
  }
  idx++;
}
elements.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 600, after: 200 },
  children: [new TextRun({ text: "Trabalho Final de Processamento de Imagens", size: 24, font: "Arial", color: "595959" })],
}));
elements.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Junho de 2026", size: 22, font: "Arial", color: "808080" })],
}));
elements.push(new Paragraph({ children: [new PageBreak()] }));

// Sumário automático
elements.push(new Paragraph({
  spacing: { after: 200 },
  children: [new TextRun({ text: "Sumário", bold: true, size: 32, font: "Arial", color: "1F3864" })],
}));
elements.push(new TableOfContents("Sumário", { hyperlink: true, headingStyleRange: "1-3" }));
elements.push(new Paragraph({ children: [new PageBreak()] }));

// Corpo: pular o título/subtítulo já processados e o bloco "## Sumário" manual
const bodyLines = [];
let skipSumario = false;
for (let j = idx; j < lines.length; j++) {
  const l = lines[j];
  if (/^##\s+Sumário/.test(l)) { skipSumario = true; continue; }
  if (skipSumario) {
    if (/^##\s/.test(l) && !/^##\s+Sumário/.test(l)) { skipSumario = false; }
    else if (/^---+\s*$/.test(l.trim())) { skipSumario = false; continue; }
    else { continue; }
  }
  bodyLines.push(l);
}

elements.push(...parseBlocks(bodyLines));

const doc = new Document({
  creator: "Claude",
  title: "Relatório — Detecção de Manobras 360",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 44, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5E8C" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "3A6B45" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Detecção de Manobras 360 — Processamento de Imagens  |  Página ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
          ],
        })],
      }),
    },
    children: elements,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log("Gerado:", OUT, "(" + buffer.length + " bytes)");
});

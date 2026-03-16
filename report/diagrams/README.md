# Diagrams

Current sources:
- `hardware_block_diagram.dot`: report-grade hardware block diagram
- `hardware_architecture.mmd`: top-level TinyNPU hardware architecture

Generated outputs:
- `hardware_block_diagram.svg`
- `hardware_architecture.svg`

Render flow:
Primary Graphviz render:
1. `dot -Tsvg report/diagrams/hardware_block_diagram.dot -o report/diagrams/hardware_block_diagram.svg`

Secondary Mermaid render:
1. `cd report/diagrams/render`
2. `npm install`
3. `npm run render`

The Mermaid render setup uses Mermaid CLI with the system Chromium browser through:
- `render/package.json`
- `render/puppeteer-config.json`

The first diagram is based on the implemented RTL hierarchy:
- `tinynpu_top`
- `control_top`
- `ubss`
- `mmio_interface`
- `control_unit`
- `instruction_memory`
- `unified_buffer`
- `systolic_array`
- `ppu`

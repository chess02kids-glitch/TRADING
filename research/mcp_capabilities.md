# MCP Capability Map

## Core Capabilities
The research agent operates via standard environment capabilities and plugins rather than specialized trading MCPs.

### Filesystem & Repository
- **Tooling**: `list_dir`, `view_file`, `write_to_file`, `replace_file_content`, `multi_replace_file_content`, `grep_search`.
- **Purpose**: Managing the research architecture, creating strategy components, building Python logic, and reading historical validation files.

### Terminal & Environment
- **Tooling**: `run_command`, `manage_task`.
- **Purpose**: Executing Python backtests, running Supabase migrations, running VectorBT screening sweeps, scheduling Cron-based daily validations, and interfacing with the Freqtrade CLI.

### Data Access & Database
- **Tooling**: Direct SQL querying via `run_command` + `psql` or `python` (psycopg/pandas). 
- **Purpose**: Writing research tracking logic to Supabase, pulling historical `ohlcv_raw` for VectorBT, reading the HAR prediction DB (`har_predictions`), and logging paper trades.

### Research Expansion
- **Tooling**: `search_web`, `read_url_content`, `read_browser_page`.
- **Purpose**: Sourcing new statistical methods, reading API documentation for updated ccxt methods, or reviewing Freqtrade advanced strategies if necessary.

## Trading / Specific MCPs
- **Current Status**: No explicitly defined trading MCP (e.g. specialized broker adapters or native backtesting MCP servers) is currently bound. All trading logic will rely on standard Python libraries (`vectorbt`, `freqtrade`, `ccxt`) running locally through `run_command`.

# Project Samarth

An intelligent Agricultural Data Q&A system built with Flask that processes natural language queries about agriculture and climate, generates insightful visualizations, and maintains comprehensive analytics.
Demo Link: https://www.loom.com/share/e86d4467a3b84add81e36f837843b90e
## Overview

Project Samarth is designed to provide easy access to agricultural insights through:
- Natural language query processing
- Interactive data visualizations
- Comprehensive agricultural analysis
- Real-time analytics tracking

## Features

### Core Capabilities
- **Smart Query Processing**: Understands complex questions about crops, rainfall, and climate
- **Multi-dimensional Analysis**: 
  - Crop production trends
  - Rainfall patterns
  - Climate impact assessment
  - State-wise comparisons
- **Interactive Visualizations**: Dynamic charts using Chart.js and Seaborn
- **Analytics Dashboard**: Tracks query patterns and system performance
- **Caching System**: Optimizes response times through intelligent caching

### Analysis Types
- Crop production analysis
- Rainfall pattern comparison
- Climate impact assessment
- State-wise agricultural overview
- Trend analysis and forecasting

## Project Structure

```
├── app.py                # Flask app and main entrypoint
├── config.env           # Configuration file (not in source control)
├── error_handler.py     # Error management system
├── query_manager.py     # Query history and analytics
├── visualizer.py        # Data visualization utilities
├── data/               # Data storage
│   ├── analytics.json  # Analytics data
│   └── query_history.json
├── templates/          # Frontend templates
│   └── index.html     # UI (chat + dashboard)
└── cache/             # Cache storage
```

## Quick Start

1. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Start the application:
   ```powershell
   python app.py
   ```

## Sample Queries

The system can process queries like:
- "Analyze rice production in Maharashtra for the last 5 years"
- "Compare rainfall between Punjab and Gujarat"
- "Show the impact of rainfall on wheat production in UP"
- "Show agricultural overview of Karnataka"

## Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment tool (venv)

### Configuration
Create a `config.env` file in the root directory with required settings. This file should not be committed to source control.

### Development Server
The app runs in debug mode for development:
```powershell
python app.py
```

## Deployment

For production deployment:
- Configure production settings in `config.env`
- Set `PRODUCTION=True` environment variable
- Use a production-grade WSGI server

A Dockerfile or Procfile can be generated for specific deployment platforms on request.

## Notes

- Maintain `config.env` securely (it's in `.gitignore`)
- Use virtual environment for development
- Follow Python coding standards (PEP 8)
- Write tests for new features

## Future Enhancements

- [ ] Enhanced data visualization options
- [ ] Machine learning-based predictions
- [ ] API documentation
- [ ] Additional data sources integration
- [ ] Performance optimization

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open for licensing. Contact maintainers to specify license preferences.
#


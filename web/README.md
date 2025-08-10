# Mini Binding - Scientist-Grade UI

A production-quality web application for ligand-protein binding prediction with calibrated probabilities, explanations, and comprehensive metrics.

## Features

- **Single & Batch Prediction**: Predict binding for individual molecules or bulk datasets
- **Calibrated Probabilities**: Temperature-scaled probabilities with abstain bands
- **Dual Explanations**: Ligand atom importance via ECFP bits and protein residue occlusion
- **Comprehensive Metrics**: ROC/PR curves, calibration plots, per-target performance
- **Offline Ready**: MSW mock mode for development without backend
- **Accessibility**: WCAG AA compliant with keyboard navigation and color-blind support

## Tech Stack

- **Frontend**: Next.js 14 App Router, TypeScript, Tailwind CSS
- **UI Components**: Radix UI primitives with shadcn/ui
- **State Management**: Zustand for global state
- **Data Tables**: TanStack Table with virtualization
- **Charts**: Recharts for metrics visualization
- **Testing**: Vitest + React Testing Library
- **Mocking**: MSW for API simulation

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Enable mock mode (when backend unavailable)
ENABLE_MOCK=1 npm run dev
```

## Environment Variables

```bash
# API base URL (default: http://localhost:8000)
NEXT_PUBLIC_API_BASE=http://localhost:8000

# Enable mock mode (1 to enable, 0 to disable)
ENABLE_MOCK=1
```

## API Endpoints

The application expects a FastAPI backend with these endpoints:

- `GET /health` - Health check and model info
- `GET /targets` - Available protein targets
- `POST /predict` - Single molecule prediction
- `POST /predict-batch` - Batch predictions
- `POST /explain` - Explanation generation
- `GET /metrics` - Model performance metrics

See `lib/schemas.ts` for complete API contracts.

## Mock Mode

When `ENABLE_MOCK=1`, the application uses MSW to simulate the backend API with realistic data distributions:

- Probabilities follow Beta(2,2) distribution
- ~5% out-of-domain molecules
- ~8% abstained predictions (default band)
- Realistic latencies and model performance

## Architecture

```
/app                    # Next.js App Router pages
  /predict             # Single prediction interface
  /batch               # Batch processing
  /metrics             # Performance analytics
  /data                # Dataset information
  /settings            # User preferences

/components            # React components
  /ui                  # Base UI components (shadcn/ui)
  TargetSelector.tsx   # Protein target selection
  SmilesInput.tsx      # SMILES input with validation
  PredictionResults.tsx # Results display

/lib                   # Core utilities
  api.ts              # API client with error handling
  schemas.ts          # Zod schemas for type safety
  store.ts            # Zustand state management
  format.ts           # Formatting and utilities

/msw                   # Mock Service Worker
  handlers.ts         # API mock implementations
  browser.ts          # Browser MSW setup
```

## Development

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Testing
npm run test
npm run test:ui

# Build for production
npm run build
```

## Production Considerations

- Set `ENABLE_MOCK=0` in production
- Configure proper CORS for API endpoints
- Use HTTPS for production deployment
- Monitor API latency and error rates
- Set up proper logging and error tracking

## Accessibility Features

- Full keyboard navigation support
- ARIA labels and semantic HTML
- Color-blind safe palette
- Focus management and screen reader support
- High contrast mode support

## Performance

- Code splitting for charts and heavy components
- Virtualized tables for large datasets
- Web Workers for CSV processing
- Image optimization and lazy loading
- Bundle size monitoring

## Security

- Input sanitization for SMILES strings
- XSS protection via React
- CSRF protection for API calls
- Content Security Policy headers
- Secure cookie configuration

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues or questions:
- Open a GitHub issue
- Check the API documentation
- Review the mock mode examples

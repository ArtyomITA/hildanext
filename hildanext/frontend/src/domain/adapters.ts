import { FrontendDataAdapter } from "./types";
import { backendAdapter } from "./backendAdapter";

export { backendAdapter };

/**
 * Active adapter: always the FastAPI backend.
 * Start the API server before using the frontend:
 *   conda run -n hilda python -m hildanext.api serve --config runs/configs/llada21_dolma_wsd_only.json
 */
export const frontendAdapter: FrontendDataAdapter = backendAdapter;

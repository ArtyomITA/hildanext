import { render, screen } from "@testing-library/react";
import { PhaseTimeline } from "../components/cards/PhaseTimeline";
import { generateWsdMetrics } from "../mocks/generators";

test("PhaseTimeline shows warmup stable and decay", () => {
  render(<PhaseTimeline metrics={generateWsdMetrics("healthy")} ladderBlocks={[1, 4, 32, 64, 512]} />);

  expect(screen.getByText("warmup")).toBeInTheDocument();
  expect(screen.getByText("stable")).toBeInTheDocument();
  expect(screen.getByText("decay")).toBeInTheDocument();
});

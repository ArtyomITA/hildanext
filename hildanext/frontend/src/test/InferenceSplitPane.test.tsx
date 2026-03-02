import { render, screen } from "@testing-library/react";
import { InferenceSplitPane } from "../features/compare/InferenceSplitPane";
import { generateInferenceScenario } from "../mocks/generators";

test("InferenceSplitPane renders AR and diffusion lanes", () => {
  const scenario = generateInferenceScenario("ar_vs_diffusion_compare", "AR vs diffusion compare", "clean");

  render(<InferenceSplitPane ar={scenario.ar} diffusion={scenario.diffusion} />);

  expect(screen.getByText("AR lane")).toBeInTheDocument();
  expect(screen.getByText("Diffusion lane")).toBeInTheDocument();
  expect(screen.getByText(/parallel drafting \+ revision/i)).toBeInTheDocument();
});

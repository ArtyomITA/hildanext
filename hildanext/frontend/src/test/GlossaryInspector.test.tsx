import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GlossaryInspector } from "../features/glossary/GlossaryInspector";

test("GlossaryInspector swaps active term content", async () => {
  render(<GlossaryInspector />);

  await userEvent.click(screen.getByRole("button", { name: "Delta" }));

  expect(screen.getByText(/token-to-token edits/i)).toBeInTheDocument();
  expect(screen.getByText(/Correzioni vere/i)).toBeInTheDocument();
});

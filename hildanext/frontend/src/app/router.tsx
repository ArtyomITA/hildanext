import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "../shell/AppShell";
import { InferencePage } from "../routes/inference/InferencePage";
import { WsdPage } from "../routes/wsd/WsdPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/wsd" replace /> },
      { path: "wsd", element: <WsdPage /> },
      { path: "inference", element: <InferencePage /> },
    ],
  },
]);

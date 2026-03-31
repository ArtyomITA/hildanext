import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "../shell/AppShell";
import { ChatPage } from "../routes/chat/ChatPage";
import { InferencePlusPage } from "../routes/inferenceplus/InferencePlusPage";
import { BenchmarkPage } from "../routes/benchmark/BenchmarkPage";
import { WsdPage } from "../routes/wsd/WsdPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: "chat", element: <ChatPage /> },
      { path: "inference", element: <Navigate to="/chat" replace /> },
      { path: "inferenceplus", element: <InferencePlusPage /> },
      { path: "benchmark", element: <BenchmarkPage /> },
      { path: "legacy/wsd", element: <WsdPage /> },
      { path: "wsd", element: <Navigate to="/legacy/wsd" replace /> },
    ],
  },
]);

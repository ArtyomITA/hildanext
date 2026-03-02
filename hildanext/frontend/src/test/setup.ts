import "@testing-library/jest-dom/vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(window, "ResizeObserver", {
  writable: true,
  value: ResizeObserverMock,
});

Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  writable: true,
  value: () => ({
    scale() {},
    clearRect() {},
    fillRect() {},
    fillText() {},
    set fillStyle(_: string) {},
    set globalAlpha(_: number) {},
    set font(_: string) {},
  }),
});

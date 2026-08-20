import "@testing-library/jest-dom/vitest"

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => undefined
}

if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = () => undefined
}

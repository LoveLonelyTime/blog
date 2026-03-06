import Typed from "typed.js";

let typed: Typed;

document.addEventListener("astro:page-load", () => {
    if (typed) typed.destroy();

    const el = document.querySelector("#motto");
    if (!el) return;

    typed = new Typed("#motto", {
        strings: ["空が^1000オレンジに^300染め上がるまで..."],
        typeSpeed: 50,
        cursorChar: '_',
    });
});

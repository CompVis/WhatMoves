const galleries = {
  transfer: [
    "pair_0032__scene_031__img_065.mp4",
    "pair_0086__scene_160__img_117.mp4",
    "pair_0127__scene_036__img_100.mp4",
    "pair_0176__scene_020__img_088.mp4",
    "pair_0208__scene_063__img_115.mp4",
    "pair_0437__scene_071__img_118.mp4",
  ],
  composite: [
    "pair_0009__scene_136__scene_161__img_024.mp4",
    "pair_0087__scene_064__scene_063__img_008.mp4",
    "pair_0109__scene_014__scene_166__img_044.mp4",
    "pair_0403__scene_150__scene_191__img_086.mp4",
    "pair_0417__scene_181__scene_141__img_052.mp4",
    "pair_0482__scene_007__scene_116__img_045.mp4",
  ],
};

const modelLabels = {
  transfer: [
    { kind: "Low-level", name: "ATI" },
    { kind: "Semantic globalized", name: "DisMo" },
    { kind: "Semantic localized", name: "Ours" },
  ],
  composite: [
    { kind: "Low-level", name: "ATI" },
    { kind: "Semantic localized", name: "Ours" },
  ],
};

const carouselAspects = {
  transfer: "3 / 2",
  composite: "3 / 1",
};

const relatedWorkFlyout = document.querySelector(".related-work-flyout");
let relatedWorkOpenTimer;
let relatedWorkCloseTimer;

relatedWorkFlyout?.addEventListener("pointerenter", () => {
  window.clearTimeout(relatedWorkCloseTimer);
  relatedWorkOpenTimer = window.setTimeout(() => {
    relatedWorkFlyout.open = true;
  }, 120);
});

relatedWorkFlyout?.addEventListener("pointerleave", () => {
  window.clearTimeout(relatedWorkOpenTimer);
  relatedWorkCloseTimer = window.setTimeout(() => {
    if (!relatedWorkFlyout.matches(":focus-within")) {
      relatedWorkFlyout.open = false;
    }
  }, 180);
});

// The page is commonly served either from the project folder or from its
// parent site directory. Try both layouts so the comparison gallery does not
// silently turn into an empty video frame.
const mediaRoots = ["../supp", "./supp"];

const carouselCard = document.querySelector("#carousel-card");
const carouselDisplay = document.querySelector(".carousel-display");
const dotsRoot = document.querySelector("#carousel-dots");
const labelsRoot = document.querySelector("#model-labels");
const tabs = Array.from(document.querySelectorAll(".gallery-tab"));
const navButtons = Array.from(
  document.querySelectorAll(".comparisons-band .carousel-button"),
);

let activeGallery = "transfer";
let activeIndex = 0;

function renderDots() {
  dotsRoot.replaceChildren();
  galleries[activeGallery].forEach((_, index) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = index === activeIndex ? "is-active" : "";
    dot.setAttribute("aria-label", `Show example ${index + 1}`);
    dot.addEventListener("click", () => {
      activeIndex = index;
      renderCarousel();
    });
    dotsRoot.append(dot);
  });
}

function renderCarousel() {
  const file = galleries[activeGallery][activeIndex];
  const labels = modelLabels[activeGallery];
  carouselDisplay.style.setProperty(
    "--carousel-aspect",
    carouselAspects[activeGallery],
  );
  const video = document.createElement("video");
  const mediaSources = mediaRoots.map(
    (root) => `${root}/${activeGallery}/${file}`,
  );
  let sourceIndex = 0;
  const tryNextMediaSource = () => {
    if (sourceIndex >= mediaSources.length - 1) return;
    sourceIndex += 1;
    video.src = mediaSources[sourceIndex];
  };

  video.addEventListener("error", tryNextMediaSource);
  video.src = mediaSources[sourceIndex];
  video.poster = `posters/${activeGallery}/${file.replace(".mp4", ".jpg")}`;
  video.setAttribute(
    "aria-label",
    `${activeGallery} comparison example ${activeIndex + 1}`,
  );
  video.controls = true;
  video.autoplay = true;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";

  labelsRoot.style.setProperty("--rows", labels.length);
  labelsRoot.replaceChildren(
    ...labels.map((label) => {
      const item = document.createElement("div");
      item.className = "model-label";

      const kind = document.createElement("span");
      kind.className = "model-kind";
      kind.textContent = label.kind;

      const name = document.createElement("strong");
      name.className = "model-name";
      name.textContent = label.name;

      item.append(kind, name);
      return item;
    }),
  );

  carouselCard.replaceChildren(video);
  renderDots();

  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
}

function move(direction) {
  const files = galleries[activeGallery];
  activeIndex = (activeIndex + direction + files.length) % files.length;
  renderCarousel();
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    activeGallery = tab.dataset.gallery;
    activeIndex = 0;
    tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    renderCarousel();
  });
});

navButtons.forEach((button) => {
  button.addEventListener("click", () => {
    move(button.dataset.action === "next" ? 1 : -1);
  });
});

const qualitativeCards = Array.from(
  document.querySelectorAll(".qualitative-card"),
);
const qualitativeButtons = Array.from(
  document.querySelectorAll(".qualitative-button"),
);
const qualitativeDots = document.querySelector(".qualitative-dots");
let qualitativeIndex = 0;

qualitativeCards.forEach((card) => {
  const video = card.querySelector("video");

  const play = () => {
    if (!card.classList.contains("is-active")) return;
    video.currentTime = 0;
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
    card.classList.add("is-playing");
  };

  const reset = () => {
    video.pause();
    video.currentTime = 0;
    card.classList.remove("is-playing");
  };

  card.addEventListener("pointerenter", play);
  card.addEventListener("pointerleave", reset);
  card.addEventListener("focus", play);
  card.addEventListener("blur", reset);
});

function renderQualitative() {
  qualitativeCards.forEach((card, index) => {
    const isActive = index === qualitativeIndex;
    const video = card.querySelector("video");
    video.pause();
    video.currentTime = 0;
    card.classList.toggle("is-active", isActive);
    card.classList.remove("is-playing");
    card.setAttribute("aria-hidden", String(!isActive));
    card.tabIndex = isActive ? 0 : -1;
  });

  qualitativeDots.replaceChildren(
    ...qualitativeCards.map((_, index) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = index === qualitativeIndex ? "is-active" : "";
      dot.setAttribute("aria-label", `Show qualitative example ${index + 1}`);
      dot.addEventListener("click", () => {
        qualitativeIndex = index;
        renderQualitative();
      });
      return dot;
    }),
  );
}

qualitativeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const direction = button.dataset.qualitativeAction === "next" ? 1 : -1;
    qualitativeIndex =
      (qualitativeIndex + direction + qualitativeCards.length) %
      qualitativeCards.length;
    renderQualitative();
  });
});

const comparisons = document.querySelector("#comparisons");
const comparisonsHint = comparisons?.querySelector(".summary-hint");

comparisons?.addEventListener("toggle", () => {
  comparisonsHint.textContent = comparisons.open
    ? "Close the gallery"
    : "Open the gallery";
});

const copyBibtexButton = document.querySelector(".copy-bibtex");
const bibtexCode = document.querySelector("#cite code");

copyBibtexButton?.addEventListener("click", async () => {
  const bibtex = bibtexCode.textContent.trim();

  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(bibtex);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = bibtex;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    const label = copyBibtexButton.querySelector("span");
    label.textContent = "Copied";
    copyBibtexButton.classList.add("is-copied");
    window.setTimeout(() => {
      label.textContent = "Copy";
      copyBibtexButton.classList.remove("is-copied");
    }, 1600);
  } catch {
    copyBibtexButton.querySelector("span").textContent = "Copy failed";
  }
});

const resultCharts = Array.from(
  document.querySelectorAll(".fresh-results .bar-chart"),
);

if ("IntersectionObserver" in window) {
  const chartObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          chartObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.35 },
  );

  resultCharts.forEach((chart) => chartObserver.observe(chart));
} else {
  resultCharts.forEach((chart) => chart.classList.add("is-visible"));
}

renderCarousel();
renderQualitative();

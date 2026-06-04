const svgDataUri = ({ label, sublabel = "", bg = "#f4f7fb", fg = "#1f2937", accent = "#2f855a" }) => {
  const safeLabel = label.replace(/[<>&"]/g, "");
  const safeSublabel = sublabel.replace(/[<>&"]/g, "");
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="320" height="220" viewBox="0 0 320 220">
      <rect width="320" height="220" rx="28" fill="${bg}"/>
      <circle cx="260" cy="40" r="52" fill="${accent}" opacity="0.18"/>
      <circle cx="56" cy="174" r="64" fill="${accent}" opacity="0.14"/>
      <rect x="52" y="60" width="216" height="98" rx="24" fill="#ffffff" opacity="0.72"/>
      <text x="160" y="108" text-anchor="middle" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="${fg}">${safeLabel}</text>
      <text x="160" y="137" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="500" fill="${fg}" opacity="0.7">${safeSublabel}</text>
    </svg>
  `;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

export const CATEGORY_IMAGES = {
  All: svgDataUri({ label: "All", sublabel: "Products", bg: "#eef8f1", accent: "#2f855a" }),
  "Fruits and Vegetables": svgDataUri({ label: "Fresh", sublabel: "Produce", bg: "#f0f9eb", accent: "#54a24b" }),
  "Meat Poultry and Seafood": svgDataUri({ label: "Protein", sublabel: "Market", bg: "#fff1f0", accent: "#d64545" }),
  Breakfast: svgDataUri({ label: "Breakfast", sublabel: "Daily", bg: "#fff8df", accent: "#d69e2e" }),
  "Chocolate and Candy": svgDataUri({ label: "Candy", sublabel: "Sweet", bg: "#fff0f6", accent: "#b83280" }),
  "Dairy and Eggs": svgDataUri({ label: "Dairy", sublabel: "Eggs", bg: "#eef6ff", accent: "#3182ce" }),
  Beverages: svgDataUri({ label: "Drinks", sublabel: "Cold", bg: "#edf8ff", accent: "#00a3c4" }),
  "Chips and Crackers": svgDataUri({ label: "Snacks", sublabel: "Crunch", bg: "#fff7ed", accent: "#dd6b20" }),
  "Ice Cream": svgDataUri({ label: "Ice", sublabel: "Cream", bg: "#f6f0ff", accent: "#805ad5" }),
};

export const HERO_DELIVERY_IMAGE = svgDataUri({
  label: "Fast",
  sublabel: "Delivery",
  bg: "#eef8f1",
  fg: "#173b2d",
  accent: "#2f855a",
});

export const WHY_IMAGES = {
  delivery: svgDataUri({ label: "20m", sublabel: "Fast", bg: "#eef8f1", accent: "#2f855a" }),
  reliable: svgDataUri({ label: "OK", sublabel: "Trusted", bg: "#eef6ff", accent: "#3182ce" }),
  prices: svgDataUri({ label: "$", sublabel: "Value", bg: "#fff8df", accent: "#d69e2e" }),
};

export const STATUS_IMAGES = {
  success: svgDataUri({ label: "OK", sublabel: "Success", bg: "#eef8f1", accent: "#2f855a" }),
  warning: svgDataUri({ label: "!", sublabel: "Warning", bg: "#fff8df", accent: "#d69e2e" }),
};

export const TRACKING_IMAGES = {
  delivery: svgDataUri({ label: "Ship", sublabel: "Track", bg: "#eef8f1", accent: "#2f855a" }),
  order: svgDataUri({ label: "Order", sublabel: "Track", bg: "#eef6ff", accent: "#3182ce" }),
};

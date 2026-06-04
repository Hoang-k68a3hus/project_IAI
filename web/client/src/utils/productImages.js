import { PRODUCT_IMAGE_BASEURL, WEB_SERVER_BASEURL } from "../api/BaseURLs";

export const getProductImageUrl = (imagePath) => {
  if (!imagePath) return `${PRODUCT_IMAGE_BASEURL}/default.jpg`;

  const fixedPath = String(imagePath)
    .replace(/\.mp4\.jpg$/i, ".jpg")
    .replace(/\.mp4$/i, ".jpg");
  if (/^https?:\/\//i.test(fixedPath)) return fixedPath;

  if (fixedPath.startsWith("/images/products/")) {
    return `${WEB_SERVER_BASEURL}${fixedPath}`;
  }

  if (fixedPath.startsWith("/")) {
    return `${WEB_SERVER_BASEURL}${fixedPath}`;
  }

  return `${PRODUCT_IMAGE_BASEURL}/${fixedPath}`;
};

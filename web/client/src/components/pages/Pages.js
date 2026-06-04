import styles from "./pages.module.css";

const Pages = ({ max, current = 1, onPageClick }) => {
  if (!max || max <= 1) return null;

  const currentPage = Number(current);
  const pages = [];
  const start = Math.max(2, currentPage - 2);
  const end = Math.min(max - 1, currentPage + 2);

  pages.push(1);
  if (start > 2) pages.push("start-ellipsis");
  for (let i = start; i <= end; i += 1) pages.push(i);
  if (end < max - 1) pages.push("end-ellipsis");
  pages.push(max);

  return (
    <div className={styles.wrapper}>
      <button
        className={styles.page}
        disabled={currentPage === 1}
        onClick={() => onPageClick(currentPage - 1)}
        aria-label="Trang truoc"
      >
        <span className="material-symbols-outlined" aria-hidden="true">
          chevron_left
        </span>
      </button>

      {pages.map((page) =>
        typeof page === "number" ? (
          <button
            key={page}
            className={`${styles.page} ${
              currentPage === page ? styles.active : ""
            }`}
            onClick={() => onPageClick(page)}
          >
            {page}
          </button>
        ) : (
          <span key={page} className={styles.ellipsis}>
            ...
          </span>
        )
      )}

      <button
        className={styles.page}
        disabled={currentPage === max}
        onClick={() => onPageClick(currentPage + 1)}
        aria-label="Trang sau"
      >
        <span className="material-symbols-outlined" aria-hidden="true">
          chevron_right
        </span>
      </button>
    </div>
  );
};

export default Pages;

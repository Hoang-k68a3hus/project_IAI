import styles from "./cartItem.module.css";

const CartItem = ({ product, updateQuantity, edit = true, id = false }) => {
  // Ép dữ liệu an toàn
  const quantity = Number(product.quantity) || 1;
  const price = Number(product.price) || 0;
  const weight = parseFloat(product.weight) || 0;
  const measurement = product.measurement || "";

  return (
    <div className={styles["wrapper"]}>
      {/* IMAGE */}
      <div className={styles["img-wrapper"]}>
        <img src={product.image} alt={product.name} />
      </div>

      {/* INFO */}
      <div className={styles["info"]}>
        <div className={styles["title"]}>{product.name}</div>

        {/* Weight x quantity */}
        <div className={styles["weight"]}>
          {(weight * quantity).toFixed(2)}
          {measurement}
        </div>

        {/* QUANTITY */}
        <div className={styles["quantity-wrapper"]}>
          {id && (
            <div className={styles["quantity"]}>#{product.product_id}</div>
          )}

          {/* Always show quantity safely */}
          <div className={styles["quantity"]}>Quantity: {quantity}</div>

          {/* ADD / REMOVE BUTTONS */}
          {edit && (
            <>
              <div
                onClick={() => updateQuantity(product, "ADD")}
                className={`${styles["btn"]} ${styles["add"]}`}
              >
                +
              </div>
              <div
                onClick={() => updateQuantity(product, "REMOVE")}
                className={`${styles["btn"]} ${styles["remove"]}`}
              >
                -
              </div>
            </>
          )}
        </div>
      </div>

      {/* PRICE */}
      <div className={styles["price"]}>
        <div className={styles["total-price"]}>
          {(price * quantity).toFixed(2)} EGP
        </div>

        {quantity > 1 && (
          <div className={styles["unit-price"]}>
            {price} EGP/{product.weight}
            {measurement}
          </div>
        )}
      </div>
    </div>
  );
};

export default CartItem;

import styles from "./wishlist.module.css";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { getWishlist } from "../../actions/auth";
import Loading from "../../components/loading/Loading";
import ProductCard from "../../components/product-card/ProductCard";
import Error from "../../components/feedback/error/Error";

const Wishlist = ({ addProductToCart }) => {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const user = useSelector((state) => state.authentication.user);

  useEffect(() => {
    const onSuccess = (wishlist) => {
      setProducts(wishlist);
      setLoading(false);
    };

    const onError = (e) => {
      // normalize axios/server errors
      const msg =
        e?.response?.data?.message || e?.message || "Failed to load wishlist";
      setError(msg);
      setLoading(false);
      // if unauthorized, redirect to login
      if (
        e?.response?.status === 401 ||
        msg.toLowerCase().includes("authorization")
      ) {
        navigate("/login");
      }
    };

    if (!user) {
      // user not logged in -> redirect to login
      setLoading(false);
      navigate("/login");
      return;
    }

    dispatch(getWishlist(onSuccess, onError));
  }, [dispatch, user, navigate]);

  if (loading) return <Loading />;

  return (
    <div className={styles["wrapper"]}>
      {error && <Error error={error} setError={setError} />}
      <div className={"heading"}>
        <h1>Wishlist</h1>
      </div>
      {!products?.length ? (
        <div className={styles["no-products"]}>
          <p>No products found in your wishlist</p>
          <Link to={"/products"} className={"btn1"}>
            Explore Products
          </Link>
        </div>
      ) : (
        <div className={styles["products-wrapper"]}>
          {products.map((product, i) => (
            <ProductCard
              product={product}
              addProductToCart={addProductToCart}
              key={i}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Wishlist;

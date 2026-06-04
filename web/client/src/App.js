import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/home/Home";
import "./shared/css/master.css";
import Navigation from "./components/navigation/Navigation";

import CartPage from "./pages/cart/Cart";
import { useEffect, useMemo, useState } from "react";
import { useSelector } from "react-redux";
import Order from "./pages/order/Order";
import Signup from "./pages/authentication/signup/Signup";
// Shipping feature removed
import Login from "./pages/authentication/login/Login";
import PrivateRoute from "./components/privete-route/PrivateRoute";
import Wishlist from "./pages/wishlist/Wishlist";
import Error401 from "./pages/errors/401/Error401";
import Error404 from "./pages/errors/404/Error404";
import Admin from "./pages/admin/default/Admin";
import AdminUpdate from "./pages/admin/products/update/default/AdminUpdate";
import AdminUpdateSuccess from "./pages/admin/products/update/success/AdminUpdateSuccess";
import AdminUpdateOrder from "./pages/admin/orders/update/AdminUpdateOrder";
import ScrollToTop from "./components/scroll-to-top/ScrollToTop";
import AdminOrders from "./pages/admin/orders/default/AdminOrders";
import AdminViewOrder from "./pages/admin/orders/id/AdminViewOrder";
import AdminNewProduct from "./pages/admin/products/new/AdminNewProduct";
import AdminShipping from "./pages/admin/shipment/default/AdminShipping";
import AdminUpdateShipping from "./pages/admin/shipment/update/AdminUpdateShipping";
import Products from "./pages/products/Products";
import ProductDetail from "./pages/product-detail/ProductDetail";
import AIDashboard from "./pages/admin/ai-dashboard/AIDashboard";

import Checkout from "./pages/checkout/checkout";
import Success from "./pages/checkout/success";
// ShipmentId removed
import OrderId from "./pages/order/id/OrderId";

const loadCartFromStorage = (cartKey) => {
  try {
    return JSON.parse(localStorage.getItem(cartKey)) || [];
  } catch (e) {
    return [];
  }
};

const App = () => {
  const authUser = useSelector((state) => state.authentication.user);

  // derive a stable user identifier fallback to 'guest'
  const userId = authUser?.user_id || authUser?._id || authUser?.id || "guest";

  const cartKey = useMemo(() => `cart_${userId}`, [userId]);
  const [cart, setCart] = useState(() => loadCartFromStorage(cartKey));
  const [cartOwnerKey, setCartOwnerKey] = useState(cartKey);

  const addProductToCart = (product) => {
    const fixedProduct = {
      ...product,
      price: Number(product.price) || 0,
      stock: product.stock ?? 100,
    };

    const productIndex = cart.findIndex(
      (cartProduct) => cartProduct.product_id === product.product_id
    );

    if (productIndex >= 0) {
      const updatedData = {
        ...cart[productIndex],
        quantity: cart[productIndex].quantity + 1,
      };
      const newArray = [...cart];
      newArray[productIndex] = updatedData;
      setCart(newArray);
    } else {
      setCart([...cart, { ...fixedProduct, quantity: 1 }]);
    }
  };

  const removeProductFromCart = (product) => {
    const productIndex = cart.findIndex(
      (cartProduct) => cartProduct.product_id === product.product_id
    );

    if (productIndex === -1) return;

    if (cart[productIndex].quantity === 1) {
      const newArr = cart.filter(
        (cartItem) => cartItem.product_id !== product.product_id
      );
      setCart(newArr);
    } else {
      const updatedData = {
        ...cart[productIndex],
        quantity: cart[productIndex].quantity - 1,
      };
      const newArray = [...cart];
      newArray[productIndex] = updatedData;
      setCart(newArray);
    }
  };

  const updateQuantity = (product, operation) => {
    if (operation === "ADD") return addProductToCart(product);

    if (operation === "REMOVE") return removeProductFromCart(product);
  };

  const [cartCount, setCartCount] = useState(0);

  useEffect(() => {
    if (cartOwnerKey !== cartKey) return;

    // save cart under the current user's cart key
    try {
      localStorage.setItem(cartKey, JSON.stringify(cart));
    } catch (e) {
      // ignore storage errors
    }

    let products = 0;
    for (const cartElement of cart) {
      products += cartElement.quantity;
    }

    setCartCount(products);
  }, [cart, cartKey, cartOwnerKey]);

  // when auth user changes (login/logout/switch), load that user's cart
  useEffect(() => {
    setCart(loadCartFromStorage(cartKey));
    setCartOwnerKey(cartKey);
  }, [cartKey]);

  return (
    <BrowserRouter>
      <ScrollToTop />
      <Navigation cartCount={cartCount} />
      <Routes>
        <Route
          path={"/"}
          element={<Home addProductToCart={addProductToCart} />}
        />
        <Route
          path={"/products"}
          element={<Products addProductToCart={addProductToCart} />}
        />
        <Route
          path={"/products/:id"}
          element={<ProductDetail addProductToCart={addProductToCart} />}
        />
        <Route
          path={"/cart"}
          element={
            <CartPage
              cart={cart}
              cartCount={cartCount}
              updateQuantity={updateQuantity}
            />
          }
        />
        <Route path={"/checkout"} element={<Checkout />} />
        <Route
          path={"/checkout/success"}
          element={<Success setCart={setCart} />}
        />
        <Route path={"/signup"} element={<Signup />} />
        <Route path={"/login"} element={<Login />} />
        <Route path={"/orders"} element={<Order />} />
        <Route path={"/orders/:id"} element={<OrderId />} />
        <Route
          path={"/wishlist"}
          element={
            <PrivateRoute
              component={<Wishlist addProductToCart={addProductToCart} />}
            />
          }
        />
        <Route
          path={"/admin"}
          element={<PrivateRoute role={"ADMIN"} component={<Admin />} />}
        />
        <Route
          path={"/admin/orders"}
          element={<PrivateRoute role={"ADMIN"} component={<AdminOrders />} />}
        />
        <Route
          path={"/admin/orders/update"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminUpdateOrder />} />
          }
        />
        <Route
          path={"/admin/products/new"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminNewProduct />} />
          }
        />
        <Route
          path={"/admin/products/update"}
          element={<PrivateRoute role={"ADMIN"} component={<AdminUpdate />} />}
        />
        <Route
          path={"/admin/products/update/success"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminUpdateSuccess />} />
          }
        />
        <Route
          path={"/admin/orders/:id"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminViewOrder />} />
          }
        />
        <Route
          path={"/admin/ai-dashboard"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AIDashboard />} />
          }
        />
        <Route
          path={"/admin/shipment"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminShipping />} />
          }
        />
        <Route
          path={"/admin/shipping"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminShipping />} />
          }
        />
        <Route
          path={"/admin/shipping/update"}
          element={
            <PrivateRoute role={"ADMIN"} component={<AdminUpdateShipping />} />
          }
        />
        <Route path={"/401"} element={<Error401 />} />
        <Route path={"/*"} element={<Error404 />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;

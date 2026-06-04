import styles from "../form.module.css";
import Authentication from "../Authentication";
import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import Error from "../../../components/feedback/error/Error";
import { authRegister } from "../../../actions/auth";
import { useDispatch } from "react-redux";

const Signup = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const [data, setData] = useState({
    first_name: "",
    last_name: "",
    email: "",
    password: "",
  });

  const [error, setError] = useState("");

  const handleChange = (e) => {
    setData({ ...data, [e.target.name]: e.target.value });
  };

  const validateEmail = (email) => {
    return String(email)
      .toLowerCase()
      .match(
        /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      );
  };

  const handleSignup = () => {
    const { first_name, last_name, email, password } = data;

    if (!first_name) return setError("Vui lòng nhập họ");
    if (!last_name) return setError("Vui lòng nhập tên");
    if (!email) return setError("Vui lòng nhập email");
    if (!validateEmail(email)) return setError("Email không hợp lệ");
    if (!password) return setError("Vui lòng nhập mật khẩu");
    if (password.length < 6)
      return setError("Mật khẩu phải có ít nhất 6 ký tự");

    const onSuccess = () => {
      navigate("/login");
    };

    const onError = (e) => {
      setError(e.message === "User already exists" ? "Email đã được sử dụng" : 
               e.message || "Lỗi máy chủ");
    };

    dispatch(
      authRegister(first_name, last_name, email, password, onSuccess, onError)
    );
  };

  const form = (
    <div className={styles["wrapper"]}>
      {error && <Error error={error} setError={setError} />}
      <div className={styles["header"]}>
        <div className={styles["title"]}>Tạo tài khoản mới ✨</div>
        <p className={styles["subtitle"]}>Tham gia cùng hàng ngàn tín đồ làm đẹp</p>
        <div className={styles["login"]}>
          Đã có tài khoản? <Link to={"/login"}>Đăng nhập</Link>
        </div>
      </div>
      <div className={styles["form"]}>
        <input
          name="first_name"
          placeholder={"Họ"}
          value={data.first_name}
          onChange={handleChange}
        />
        <input
          name="last_name"
          placeholder={"Tên"}
          value={data.last_name}
          onChange={handleChange}
        />
        <input
          name="email"
          placeholder={"Email của bạn"}
          value={data.email}
          onChange={handleChange}
          type={"email"}
        />
        <input
          name="password"
          placeholder={"Mật khẩu (tối thiểu 6 ký tự)"}
          value={data.password}
          onChange={handleChange}
          type={"password"}
        />
        <button onClick={handleSignup} className={"btn1"}>
          Đăng ký
        </button>
      </div>
    </div>
  );

  return <Authentication data={form} />;
};

export default Signup;

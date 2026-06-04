import styles from '../form.module.css';
import {Link, useNavigate} from "react-router-dom";
import Authentication from "../Authentication";
import {useState} from "react";
import Error from "../../../components/feedback/error/Error";
import {authLogin} from "../../../actions/auth";
import {useDispatch} from "react-redux";

const Login = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const [data, setData] = useState({
        email: "",
        password: ""
    });

    const [error, setError] = useState("");

    const handleChange = (e) => {
        setData({...data, [e.target.name]: e.target.value});
    }

    const handleLogin = () => {
        const {email, password} = data;

        if (!email)
            return setError("Vui lòng nhập email");

        if (!password)
            return setError("Vui lòng nhập mật khẩu");

        if (!validateEmail(email))
            return setError("Email không hợp lệ");

        if (password.length < 6)
            return setError("Mật khẩu phải có ít nhất 6 ký tự");

        const onSuccess = () => {
            navigate('/');
        }

        const onError = (e) => {
            setError(e.message === "User not found" ? "Tài khoản không tồn tại" : 
                     e.message === "Wrong password" ? "Mật khẩu không đúng" : e.message);
        }

        dispatch(authLogin(email, password, onSuccess, onError));
    }

    const validateEmail = (email) => {
        return String(email)
            .toLowerCase()
            .match(
                /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
            );
    };

    const form =
        <div className={styles['wrapper']}>
            {error && <Error error={error} setError={setError}/>}
            <div className={styles['header']}>
                <div className={styles['title']}>Chào mừng trở lại! 💄</div>
                <p className={styles['subtitle']}>Đăng nhập để khám phá thế giới mỹ phẩm</p>
                <div className={styles['login']}>Chưa có tài khoản? <Link to={'/signup'}>Đăng ký ngay</Link></div>
            </div>
            <div className={styles['form']}>
                <input onChange={(e) => handleChange(e)} name={'email'} value={data.email} placeholder={'Email của bạn'}
                       type={'email'}/>
                <input onChange={(e) => handleChange(e)} name={'password'} value={data.password}
                       placeholder={'Mật khẩu'} type={'password'}/>
                <button onClick={handleLogin} className={'btn1'}>Đăng nhập</button>
            </div>
        </div>

    return <Authentication data={form}/>
}

export default Login;
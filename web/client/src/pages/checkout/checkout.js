import styles from './checkout.module.css';
import '../.././shared/css/master.css';
import {useState, useRef} from "react";
import {useDispatch, useSelector} from "react-redux";
import Error from "../../components/feedback/error/Error";
import {postOrder} from "../../actions/orders";

const Checkout = () => {
    const [error, setError] = useState('');
    const fname = useRef();
    const lname = useRef();
    const email = useRef();
    const phone = useRef();
    const city = useRef();
    const area = useRef();
    const street = useRef();
    const building_number = useRef();
    const floor = useRef();
    const apartment_number = useRef();
    const dispatch = useDispatch();

    const cart = useSelector(state => state.products.cart_validation);

    const handleCheckout = () => {

        if (!fname.current.value)
            return setError("Enter a first name");
        if (!lname.current.value)
            return setError("Enter a last name");
        if (!email.current.value)
            return setError("Enter an email address");
        if (!validateEmail(email.current.value))
            return setError("Enter a valid email address");
        if (!phone.current.value)
            return setError("Enter a phone number");
        if (!validatePhone(phone.current.value))
            return setError("Enter a valid phone number");
        if (!city.current.value)
            return setError("Enter a city");
        if (!area.current.value)
            return setError("Enter an area");
        if (!street.current.value)
            return setError("Enter a street");
        if (!building_number.current.value)
            return setError("Enter a building number");
        if (isNaN(building_number.current.value))
            return setError("Enter a valid building number");
        if (!floor.current.value)
            return setError("Enter a floor");
        if (isNaN(floor.current.value))
            return setError("Enter a valid building number");
        if (!apartment_number.current.value)
            return setError("Enter an apartment number");
        if (isNaN(apartment_number.current.value))
            return setError("Enter a valid building number");


        const onSuccess = (url) => {
            window.location.href = url;
        }

        const onError = (e) => {
            setError(e.message);
        }


        const data = {

            name: {
                first: fname.current.value,
                last: lname.current.value,
            },
            email: email.current.value,
            phone_number: phone.current.value,
            address: {
                country: 'Vietnam',
                city: city.current.value,
                area: area.current.value,
                street: street.current.value,
                building_number: building_number.current.value,
                floor: floor.current.value,
                apartment_number: apartment_number.current.value
            }
        };

        dispatch(postOrder(cart.token, data, onSuccess, onError));

    }
    const validatePhone = (phone) => {
        // Vietnam phone numbers: 03x, 05x, 07x, 08x, 09x (10 digits)
        return String(phone)
            .toLowerCase()
            .match(
                /^(03|05|07|08|09)[0-9]{8}$/
            );
    };

    const validateEmail = (email) => {
        return String(email)
            .toLowerCase()
            .match(
                /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
            );
    };

    return (
        <div className={styles['wrapper']}>
            {error && <Error error={error} setError={setError}/>}
            <div className={'heading-wrapper'}>
                <h1 className={'heading'}>Checkout</h1>
            </div>

            <div className={styles['form']}>
                <input ref={fname} type="text" placeholder="Tên"/>
                <input ref={lname} type="text" placeholder="Họ"/>

                <input ref={email} type="text" placeholder="Email"/>
                <input ref={phone} type="text" placeholder="Số điện thoại (VN: 03x, 05x...)"/>
                <input ref={city} type="text" placeholder="Thành phố"/>

                <input ref={area} type="text" placeholder="Quận/Huyện"/>
                <input ref={street} type="text" placeholder="Đường phố"/>
                <input ref={building_number} type="text" placeholder="Số nhà"/>

                <input ref={floor} type="text" placeholder="Tầng"/>
                <input ref={apartment_number} type="text" placeholder="Số căn hộ"/>
            </div>

            <div className={styles['total']}>
                <div className={styles["total-text"]}>Tổng tiền:</div>
                <div className={styles['total-amount']}>{cart.total.toLocaleString('vi-VN')} VNĐ</div>
            </div>

            <div className={styles['total-wrapper']}>
                <button onClick={handleCheckout} className={'btn1'}>Đặt hàng</button>
            </div>

        </div>
    )
}
export default Checkout;
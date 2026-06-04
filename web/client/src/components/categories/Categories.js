import styles from './categories.module.css';
import {Link} from "react-router-dom";
import {CATEGORY_IMAGES} from "../../shared/placeholders";

const Categories = () => {
    const categories = [
        {
            display: "All",
            img: CATEGORY_IMAGES.All
        },
        {
            display: "Fruits and Vegetables",
            img: CATEGORY_IMAGES["Fruits and Vegetables"]
        },
        {
            display: "Meat Poultry and Seafood",
            img: CATEGORY_IMAGES["Meat Poultry and Seafood"]
        },
        {
            display: "Breakfast",
            img: CATEGORY_IMAGES.Breakfast
        },
        {
            display: "Chocolate and Candy",
            img: CATEGORY_IMAGES["Chocolate and Candy"]
        },
        {
            display: "Dairy and Eggs",
            img: CATEGORY_IMAGES["Dairy and Eggs"]
        },
        {
            display: "Beverages",
            img: CATEGORY_IMAGES.Beverages
        },
        {
            display: "Chips and Crackers",
            img: CATEGORY_IMAGES["Chips and Crackers"]
        },
        {
            display: "Ice Cream",
            img: CATEGORY_IMAGES["Ice Cream"]
        }
    ]

    return (
        <div className={styles['categories']}>
            <div className={`${styles['categories-scroll']}`}>
                {categories.map((item, i) =>
                    <Link key={i} to={`/products?category=${item.display}`} className={styles['category']}>
                        <div>{item.display}</div>
                        <img src={item.img} alt={item.display}/>
                    </Link>
                )}
            </div>
        </div>
    );
}

export default Categories;

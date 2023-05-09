import User from "../models/user.js";
const HomePage = async (req, res) => {
    const name = req.query.name;
    const password = req.query.password;
    try {
        const user = await User.findOne({ name, password }).exec();
        console.log(user)
        if (user) {
            if (req.cookies.myCookie) {
                const myCookie = req.cookies.myCookie;
                console.log('Cookie already set');
                res.render('home.ejs', {data:myCookie })
            }
            else{
            res.cookie('myCookie', name,{ path: '/' });
            const myCookie = req.cookies.myCookie;
                console.log('cookie assigned ' + myCookie);
                res.render('home.ejs', {data:myCookie })
                }
            
        }
    }
    catch (err) {
        console.log(err);
    }
    res.sendFile('home.html', { root: 'source/Template' });
}
export default HomePage;
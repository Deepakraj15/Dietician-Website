const Login = (req,res) => {
    res.sendFile('login.html', { root: 'source/Template' });
}
export default Login;
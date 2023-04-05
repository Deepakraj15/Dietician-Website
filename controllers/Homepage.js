const HomePage = (req,res) => {
    res.sendFile('home.html', { root: 'source/Template' });
}
export default HomePage;
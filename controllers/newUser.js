const newUser = (req, res) =>
{
    res.sendFile('newUser.html', { root: 'source/Template' });
}

export default newUser;



const x = async () => {
    try {
        throw new Error('hello')
    } catch (e)
    {
        throw new Error('hi')
    }
    
}


const main = async () => {
    try {
        await x();
    } catch(e)
    {
        console.log(e.message)
    }
    
}

main();
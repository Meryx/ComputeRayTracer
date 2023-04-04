const createSampler = ({device, addressModeU, addressModeV, magFilter, minFilter, mipmapFilter, maxAnisotropy}) => {
    const sampler = device.createSampler({
        addressModeU,
        addressModeV,
        magFilter,
        minFilter,
        mipmapFilter,
        maxAnisotropy,
    });
    return sampler
}

export default createSampler;
const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        const isCancer = confidenceScore > 50;
        const classes = ['Non-Cancer' , 'Cancer'];
        
        const label = classes[isCancer ? 1 : 0];

        let explanation, suggestion;
 
        if(isCancer) {
            explanation = 'Telah Terdeteksi Kanker Kulit Pada Gambar Yang Dikirimkan'
            suggestion = 'Segera Konsultasi Ke Dokter Spesialis Kulit'
        }
        else {
            explanation = 'Tidak Terdeteksi Kanker Kulit Pada Gambar Yang Dikirimkan'
            suggestion = 'Tetap Jaga Kesehatan Kulitmu Yaa!'
        }
 
    return { label, confidenceScore, explanation, suggestion };

    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}
 
module.exports = predictClassification;
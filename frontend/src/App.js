import './App.css';
import axios from 'axios';
import { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    input1: '',
    input2: ''
  });


  const [percentage, setPercentage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/test', formData);
      console.log('Form submitted successfully:', response.data);
      setFormData({
        input1: '',
        input2: ''
      });


      if(response){
        setPercentage(response.data);
      }
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  return (
    <div className="App">
      <form onSubmit={handleSubmit}>
        <input className='input1' type="text" name="input1" value={formData.input1} onChange={handleChange} placeholder="Enter value for input 1" />
        <input className='input2' type="text" name="input2" value={formData.input2} onChange={handleChange} placeholder="Enter value for input 2" />
        <button type="submit">Submit</button>
      </form>
      {percentage && <div className='result'>{percentage}</div>}
    </div>
  );
}

export default App;

import cherrypy
import json
from data_controller import YourDataAnalyzer

class DoubleArrayAPI:


    def __init__(self):

        self.input_data = None



    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()

    def load_data(self):

        try:

            self.input_data = cherrypy.request.json  # Store the input data

            return {"message": "Input data stored"}
        
        except Exception as e:

            cherrypy.response.status = 400

            return {"error": str(e)}
        

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    @cherrypy.tools.allow(methods=['POST'])
    @cherrypy.expose

    def r2(self):

        try:

            if self.input_data is None:

                raise ValueError("Input data not provided")

            input_array = self.input_data.get("input_array", [])

            controller  = YourDataAnalyzer(input_array)

            result      = controller.analyze()


            return {"result": result}
        
        
        except Exception as e:

            cherrypy.response.status = 400

            return {"error": str(e)}    


if __name__ == '__main__':
    
    cherrypy.quickstart(DoubleArrayAPI())
